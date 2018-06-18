This is a repo with some experiments on text VAE.

Random thoughts:
- Try [discrete latent code](https://arxiv.org/abs/1804.00104). Try [other distributions](https://arxiv.org/abs/1805.08498) for latent code.
- What if encoder produces matrix and we transform noise with this matrix? Or transform word embeddings? Will it be equivalent to the current approach? For any distribution?
- How do we do clustering with GANs?
- Try k-means/GMM on top of TF-IDF features as a baseline.
- Can we force independence on z by somehow minimizing covariances?
- How can we formulate "disentaglement" more formally? Is it just variable independence?
- Yes/no glasses, yes/no smiling and all of them are different. For each/some normal attribute we should keep a binary attribute, denoting its off/on state.
- Group latent dimensions into inter-dependent blocks and allow covariance in each block. Our covariance matrix is a block matrix.
