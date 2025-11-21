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
    import pdb; pdb.set_trace()
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

if __name__ == '__main__':
    main1()
    