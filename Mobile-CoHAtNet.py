def depthwise_separable_conv(inp, oup, image_size, downsample=False):
    stride = 2 if downsample else 1
    return nn.Sequential(
        nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),  # Depthwise
        nn.BatchNorm2d(inp),
        nn.GELU(),
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),                    # Pointwise
        nn.BatchNorm2d(oup),
        nn.GELU(),
    )


class PreNorm(nn.Module):
    def __init__(self, dim, fn, norm):
        super().__init__()
        self.norm = norm(dim)
        self.fn   = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class SE(nn.Module):
    def __init__(self, inp, oup, expansion=0.25):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(oup, int(inp*expansion), bias=False),
            nn.GELU(),
            nn.Linear(int(inp*expansion), oup, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class MBConv(nn.Module):
    def __init__(self, inp, oup, image_size, downsample=False, expansion=4):
        super().__init__()
        self.downsample = downsample
        stride = 2 if downsample else 1
        hidden_dim = int(inp * expansion)

        if self.downsample:
            self.pool = nn.MaxPool2d(3, 2, 1)
            self.proj = nn.Conv2d(inp, oup, 1, 1, 0, bias=False)

        if expansion == 1:
            self.conv = nn.Sequential(
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.GELU(),
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                nn.Conv2d(inp, hidden_dim, 1, stride, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.GELU(),
                nn.Conv2d(hidden_dim, hidden_dim, 3, 1, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.GELU(),
                SE(inp, hidden_dim),
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )

        self.conv = PreNorm(inp, self.conv, nn.BatchNorm2d)

    def forward(self, x):
        if self.downsample:
            return self.proj(self.pool(x)) + self.conv(x)
        else:
            return x + self.conv(x)


class Attention(nn.Module):
    def __init__(self, inp, oup, image_size, heads=4, dim_head=32, dropout=0.):
        super().__init__()
        inner_dim  = dim_head * heads
        project_out = not (heads == 1 and dim_head == inp)

        self.ih, self.iw = image_size
        self.heads = heads
        self.scale = dim_head**-0.5

        # relative position bias
        self.relative_bias_table = nn.Parameter(torch.zeros((2*self.ih-1)*(2*self.iw-1), heads))
        coords = torch.meshgrid((torch.arange(self.ih), torch.arange(self.iw)))
        coords = torch.flatten(torch.stack(coords), 1)
        relative_coords = coords[:, :, None] - coords[:, None, :]
        relative_coords[0] += self.ih - 1
        relative_coords[1] += self.iw - 1
        relative_coords[0] *= (2*self.iw - 1)
        relative_coords = rearrange(relative_coords, 'c h w -> h w c')
        relative_index  = relative_coords.sum(-1).flatten().unsqueeze(1)
        self.register_buffer("relative_index", relative_index)

        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(inp, inner_dim*3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, oup),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        relative_bias = self.relative_bias_table.gather(
            0, self.relative_index.repeat(1, self.heads)
        )
        relative_bias = rearrange(relative_bias, '(h w) c -> 1 c h w',
                                  h=self.ih*self.iw, w=self.ih*self.iw)
        dots = dots + relative_bias

        attn = self.attend(dots)
        out  = torch.matmul(attn, v)
        out  = rearrange(out, 'b h n d -> b n (h d)')
        out  = self.to_out(out)
        return out


class HAttention(nn.Module):
    def __init__(self, inp, oup, image_size, heads=4, dim_head=32, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == inp)

        self.ih, self.iw = image_size
        self.heads       = heads
        self.scale       = dim_head**-0.5

        self.relative_bias_table = nn.Parameter(torch.zeros((2*self.ih-1)*(2*self.iw-1), heads))
        coords = torch.meshgrid((torch.arange(self.ih), torch.arange(self.iw)))
        coords = torch.flatten(torch.stack(coords), 1)
        relative_coords = coords[:, :, None] - coords[:, None, :]
        relative_coords[0] += self.ih - 1
        relative_coords[1] += self.iw - 1
        relative_coords[0] *= (2*self.iw - 1)
        relative_coords = rearrange(relative_coords, 'c h w -> h w c')
        relative_index  = relative_coords.sum(-1).flatten().unsqueeze(1)
        self.register_buffer("relative_index", relative_index)

        self.attend = nn.Softmax(dim=-1)
        self.to_qk  = nn.Linear(inp, inner_dim*2, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(oup, oup),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x, mbconv):
        qk = self.to_qk(x).chunk(2, dim=-1)
        q, k = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qk)
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        relative_bias = self.relative_bias_table.gather(
            0, self.relative_index.repeat(1, self.heads)
        )
        relative_bias = rearrange(relative_bias, '(h w) c -> 1 c h w',
                                  h=self.ih*self.iw, w=self.ih*self.iw)
        dots = dots + relative_bias

        attn = self.attend(dots)
        v = rearrange(mbconv, 'b c ih iw -> b (ih iw) c')
        v = rearrange(v, 'b n (h c) -> b h n c', h=self.heads)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        return out


class Transformer(nn.Module):
    def __init__(self, inp, oup, image_size, heads=4, dim_head=32, downsample=False, dropout=0.):
        super().__init__()
        hidden_dim = int(inp*4)
        self.ih, self.iw = image_size
        self.downsample  = downsample

        if self.downsample:
            self.pool = nn.MaxPool2d(3, 2, 1)
            self.proj = nn.Conv2d(inp, oup, 1, 1, 0, bias=False)

        self.attn = Attention(inp, oup, image_size, heads, dim_head, dropout)
        self.ff   = FeedForward(oup, hidden_dim, dropout)

        self.attn = nn.Sequential(
            Rearrange('b c ih iw -> b (ih iw) c'),
            PreNorm(inp, self.attn, nn.LayerNorm),
            Rearrange('b (ih iw) c -> b c ih iw', ih=self.ih, iw=self.iw)
        )

        self.ff = nn.Sequential(
            Rearrange('b c ih iw -> b (ih iw) c'),
            PreNorm(oup, self.ff, nn.LayerNorm),
            Rearrange('b (ih iw) c -> b c ih iw', ih=self.ih, iw=self.iw)
        )

    def forward(self, x):
        if self.downsample:
            x = self.proj(self.pool(x)) + self.attn(self.pool(x))
        else:
            x = x + self.attn(x)
        x = x + self.ff(x)
        return x


class HTransformer(nn.Module):
    def __init__(self, inp, oup, image_size, heads=4, dim_head=32, downsample=False, dropout=0.):
        super().__init__()
        hidden_dim    = int(inp*4)
        self.ih, self.iw = image_size
        self.downsample  = downsample
        self.layer_norm  = nn.LayerNorm(inp)
        self.MBConv      = MBConv(inp, oup, image_size, downsample)

        if self.downsample:
            self.pool1 = nn.MaxPool2d(3, 2, 1)
            self.pool2 = nn.MaxPool2d(3, 2, 1)
            self.proj  = nn.Conv2d(inp, oup, 1, 1, 0, bias=False)

        self.attn = HAttention(inp, oup, image_size, heads, dim_head, dropout)
        self.ff   = FeedForward(oup, hidden_dim, dropout)

        self.ff   = nn.Sequential(
            Rearrange('b c ih iw -> b (ih iw) c'),
            PreNorm(oup, self.ff, nn.LayerNorm),
            Rearrange('b (ih iw) c -> b c ih iw', ih=self.ih, iw=self.iw)
        )

    def forward(self, x):
        mbconv = self.MBConv(x)
        if self.downsample:
            pool1  = self.pool2(x)
            pool1  = rearrange(pool1, 'b c ih iw -> b (ih iw) c')
            norm1  = self.layer_norm(pool1)
            attn1  = self.attn(norm1, mbconv)
            attn1  = rearrange(attn1, 'b (ih iw) c -> b c ih iw', ih=self.ih, iw=self.iw)
            out1   = self.proj(self.pool1(x)) + attn1
        else:
            xx     = rearrange(x, 'b c ih iw -> b (ih iw) c')
            norm1  = self.layer_norm(xx)
            attn1  = self.attn(norm1, mbconv)
            attn1  = rearrange(attn1, 'b (ih iw) c -> b c ih iw', ih=self.ih, iw=self.iw)
            out1   = x + attn1
        x = out1 + self.ff(out1)
        return x


class CoHAtNet(nn.Module):
    def __init__(self, image_size, in_channels, num_blocks, channels,
                 num_classes=1000, block_types=['C','C','H','H']):
        super().__init__()
        ih, iw = image_size
        block = {'C': MBConv, 'T': Transformer, 'H': HTransformer}

        # Stages
        self.s0 = self._make_layer(depthwise_separable_conv, in_channels, channels[0], num_blocks[0], (ih//2, iw//2))
        self.s1 = self._make_layer(block[block_types[0]], channels[0], channels[1], num_blocks[1], (ih//4, iw//4))
        self.s2 = self._make_layer(block[block_types[1]], channels[1], channels[2], num_blocks[2], (ih//8, iw//8))
        self.s3 = self._make_layer(block[block_types[2]], channels[2], channels[3], num_blocks[3], (ih//16, iw//16))
        self.s4 = self._make_layer(block[block_types[3]], channels[3], channels[4], num_blocks[4], (ih//32, iw//32))

        self.pool = nn.AvgPool2d(ih//32, 1)

        # -------- NEW: IMU Embedding MLP --------
        self.imu_fc = nn.Sequential(
            nn.Linear(6, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU()
        )

        # After the pool, we get [B, channels[-1]].
        # The code does cat(x, x), so dimension doubles => 2*channels[-1].
        # Then we add +128 for IMU embedding => total_in
        total_in  = (channels[-1]*2) + 128
        hidden_dim = 1024

        self.fc1  = nn.Linear(total_in, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2  = nn.Linear(hidden_dim, hidden_dim//2)
        self.fc3  = nn.Linear(hidden_dim//2, 7)  # pose: x, y, z, qx, qy, qz, qw

    def forward(self, x, imu=None):
        # 1) Forward pass through each stage
        x = self.s0(x)
        x = self.s1(x)
        x = self.s2(x)
        x = self.s3(x)
        x = self.s4(x)

        # 2) Global pooling => shape [B, channels[-1]]
        x = self.pool(x).view(-1, x.shape[1])

        # 3) Original code: we cat(x, x), doubling channels
        x = torch.cat((x, x), dim=1)  # shape [B, 2*channels[-1]]

        # 4) If we have IMU data, embed it and cat
        if imu is not None:
            imu_embed = self.imu_fc(imu)            # shape [B, 128]
            x = torch.cat((x, imu_embed), dim=1)    # shape [B, 2*channels[-1] + 128]

        # 5) Final MLP layers
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x

    def _make_layer(self, block, inp, oup, depth, image_size):
        layers = nn.ModuleList([])
        for i in range(depth):
            if i == 0:
                layers.append(block(inp, oup, image_size, downsample=True))
            else:
                layers.append(block(oup, oup, image_size))
        return nn.Sequential(*layers)


def cohatnet_mobile():
    num_blocks = [1, 1, 2, 2, 1]            # Reduced depth
    channels   = [32, 48, 96, 192, 384]     # Reduced width
    return CoHAtNet(
        image_size=(256, 256),
        in_channels=3,
        num_blocks=num_blocks,
        channels=channels,
        num_classes=7
    )
