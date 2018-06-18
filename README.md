Какие признаки можно доклеивать:
- количество сообщений
- длина диалога (временная)
- количество слов, символов
- среднее количество слов в сообщении
- доля специфичных слов (вытаскиваем каким-нибудь TF-IDF, BigARTM)
- наличие отдельных специфичных слов (вытаскиваем каким-нибудь TF-IDF, BigARTM)
- латентный код из автоэнкодера

Что можно улучшить/поменять:
- А что если латентный код будет полностью дискретный? Куча переменных с дискретными штуками. Какой там будет prior? Сможем ли мы хорошо считать KL для них? Попробовать взять с размерностью 1. Тогда будет чисто кластеризация. Зато информация о кластере никуда больше не утечет.
- А что если мы будем делать VAE следующим образом. Энкодер выдает матрицу размера 64x64. Мы сэмплим шум размера 64 и преобразовываем его этой матрицей. Будет ли это эквивалентно существующему подходу? Или преобразовывать матрицей эмбеддинги слов?
- Можно ли сделать кластеризацию на K кластеров ганами с 1 генератором и K дискриминаторами? Или наоборот? Или как-то еще? Кажется, что если заставить нейронку как-то насильно отделять предложения, то должно получиться лучше.
- Неужели кластеризация текстов — это такая тяжелая штука. Надо попробовать K-means/GMM на TF-IDF признаках.
- Попробовать воспроизвести ту статью про TextVAE с CNN, но только beta-VAE.
- Вместо KL на z использовать минимизацию энтропии.
- Can we force independence on z by just minimizing covariances?
- How can we formulate "disentaglement" more formally? Can we do it through variable independence?
- Yes/no glasses, yes/no smiling and all of them are different. For each/some normal attribute we should keep a binary attribute, denoting its off/on state. Or not?
- Group latent dimensions into inter-dependent blocks and allow correlation in each block. Our covariance matrix is a blocked matrix.
- Progressively increase noise/KL loss. So that we progressively move from AE to VAE and decoder learns to condition on z.
