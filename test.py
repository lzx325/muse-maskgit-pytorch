import torch
from muse_maskgit_pytorch import VQGanVAE, VQGanVAETrainer, MaskGitTransformer, MaskGit

def main1():
    vae = VQGanVAE(
        dim = 256,
        codebook_size = 65536
    )

    transformer = MaskGitTransformer(
        num_tokens = 65536,       # must be same as codebook size above
        seq_len = 256,            # must be equivalent to fmap_size ** 2 in vae
        dim = 512,                # model dimension
        depth = 8,                # depth
        dim_head = 64,            # attention head dimension
        heads = 8,                # attention heads,
        ff_mult = 4,              # feedforward expansion factor
        t5_name = 't5-small',     # name of your T5
    )

    images = torch.randn(4, 3, 256, 256)

    fmap, indices, vq_aux_loss = vae.encode(images) # fmap: (B, 2048, 16, 16), indices (B, 16, 16)
    print(vae.quantizer.num_codebooks, vae.quantizer.codebook_dim, fmap.shape, indices.shape, vq_aux_loss)
    

    print(transformer)

    base_maskgit = MaskGit(
        vae = vae,                 # vqgan vae
        transformer = transformer, # transformer
        image_size = 256,          # image size
        cond_drop_prob = 0.25,     # conditional dropout, for classifier free guidance
    )

    # ready your training text and images

    texts = [
        'a child screaming at finding a worm within a half-eaten apple',
        'lizard running across the desert on two feet',
        'waking up to a psychedelic landscape',
        'seashells sparkling in the shallow waters'
    ]

    # feed it into your maskgit instance, with return_loss set to True
    loss = base_maskgit(
        images,
        texts = texts
    )

    images = base_maskgit.generate(texts = [
        'a whale breaching from afar',
        'young girl blowing out candles on her birthday cake',
        'fireworks with blue and green sparkles'
    ], cond_scale = 3.) # conditioning scale for classifier free guidance

    print(images.shape) # (3, 3, 256, 256)

def main2():
    from muse_maskgit_pytorch.muse_maskgit_pytorch import get_mask_subset_prob
    mask = torch.tensor(
        [
            [0,1,0,1,1,1],
            [1,1,0,0,0,0],
        ], 
        dtype = torch.bool
    )
    scores = torch.tensor(
        [
            [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
            [0.5, 0.6, 0.4, 0.3, 0.2, 0.1]
        ],
        dtype = torch.float32
    )
    mask_subset = get_mask_subset_prob(mask, 0.5, scores = scores)
    print(mask_subset)

def main3():
    from einops import rearrange

    vae = VQGanVAE(
        dim = 256,
        codebook_size = 65536
    )

    transformer = MaskGitTransformer(
        num_tokens = 65536,       # must be same as codebook size above
        seq_len = 256,            # must be equivalent to fmap_size ** 2 in vae
        dim = 512,                # model dimension
        depth = 8,                # depth
        dim_head = 64,            # attention head dimension
        heads = 8,                # attention heads,
        ff_mult = 4,              # feedforward expansion factor
        t5_name = 't5-small',     # name of your T5
        cross_attend = False
    )

    base_maskgit = MaskGit(
        vae = vae,                 # vqgan vae
        transformer = transformer, # transformer
        image_size = 256,          # image size
        cond_drop_prob = 0.25,     # conditional dropout, for classifier free guidance
    )

    images = torch.randn(4, 3, 256, 256)
    _, indices, _ = vae.encode(images)

    indices = rearrange(indices, 'b ... -> b (...)')
    
    indices_m = indices.scatter(1, torch.IntTensor([[0,1,2,3,4]]*4), base_maskgit.mask_id)

    indices_um = base_maskgit.unmask(indices_m, timesteps = 6)
    print(indices_um[:,:6])
    import pdb; pdb.set_trace()

if __name__ == '__main__':
    main3()
    