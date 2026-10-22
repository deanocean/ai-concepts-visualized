/* =========================================================
   概念清單（依 JDLA G検定 シラバス2024 第1.4版）
   - area：所屬分區 id
   - sec：シラバス節號
   - en：英文名稱（各語言共用，標題下方顯示）
   - file：已完成的概念填上模組檔路徑；沒有 file ＝「準備中」
   - i18n：true ＝ 文字已拆到 locales/<lang>/<id>.js（中日文都有）；沒有 ＝ 文字仍寫在概念檔裡（只有中文）
   - added：true ＝ 使用者原本清單沒有、依シラバス補充的項目
   - deps：先載入的共用檔（assets/nn.js：小型神經網路、即時訓練迴圈、折線圖；生成模型、強化學習、word2vec 用）
   分區與概念的顯示名稱放在 locales/<lang>/_core.js（area.<id>、name.<id>、desc.<id>）。
   新增概念的步驟：
     1. 寫 concepts/<id>.js（呼叫 Atlas.define("<id>", {...})，文字用 ctx.t(key)）
     2. 寫 locales/zh/<id>.js 與 locales/ja/<id>.js，並在兩個 _core.js 加上 name.<id>、desc.<id>
     3. 在下面對應項目加上 file: "concepts/<id>.js", i18n: true
     4. 執行 node tools/check-i18n.mjs
   ========================================================= */
const NN = ["assets/nn.js"];
Atlas.registry(
  [
    { id: "ml", sec: "§7–10" },
    { id: "learn", sec: "§12–16・19" },
    { id: "cnn", sec: "§18・20・21" },
    { id: "seq", sec: "§22・23・27" },
    { id: "gen", sec: "§24・30" },
    { id: "rl", sec: "§9・29" },
  ],
  [
    /* 機械學習基礎 */
    { id: "kernel", area: "ml", sec: "§7",  en: "SVM & Kernel Trick", added: true, file: "concepts/svm.js", i18n: true },
    { id: "kmeans", area: "ml", sec: "§8",  en: "k-means Clustering", added: true, file: "concepts/kmeans.js", i18n: true },
    { id: "pca", area: "ml", sec: "§8",  en: "Principal Component Analysis", added: true, file: "concepts/pca.js", i18n: true },
    { id: "roc", area: "ml", sec: "§10", en: "ROC Curve & AUC", added: true, file: "concepts/roc.js", i18n: true },

    /* 神經網路怎麼學習 */
    { id: "gd", area: "learn", sec: "§16", en: "Gradient Descent & Learning Rate", file: "concepts/gradient-descent.js", i18n: true },
    { id: "sgd", area: "learn", sec: "§16", en: "SGD & Mini-batch", added: true, file: "concepts/sgd.js", i18n: true },
    { id: "optim", area: "learn", sec: "§16", en: "Momentum & Adam", added: true, file: "concepts/optim.js", i18n: true },
    { id: "backprop", area: "learn", sec: "§15", en: "Backpropagation & Chain Rule", file: "concepts/backprop.js", i18n: true },
    { id: "vanish", area: "learn", sec: "§12・15", en: "Vanishing Gradient", file: "concepts/vanishing.js", i18n: true },
    { id: "act", area: "learn", sec: "§12", en: "Activation Functions", file: "concepts/activation.js", i18n: true },
    { id: "loss", area: "learn", sec: "§13", en: "Loss Functions", added: true, file: "concepts/loss.js", i18n: true },
    { id: "overfit", area: "learn", sec: "§10・14", en: "Overfitting & Regularization", file: "concepts/overfitting.js", i18n: true },
    { id: "dropout", area: "learn", sec: "§14", en: "Dropout", file: "concepts/dropout.js", i18n: true },
    { id: "norm", area: "learn", sec: "§19", en: "Normalization Layers", file: "concepts/norm.js", i18n: true },

    /* 卷積與影像 */
    { id: "conv", area: "cnn", sec: "§18", en: "Convolution", file: "concepts/convolution.js", i18n: true },
    { id: "pool", area: "cnn", sec: "§20", en: "Pooling", file: "concepts/pooling.js", i18n: true },
    { id: "dilated", area: "cnn", sec: "§18", en: "Receptive Field & Dilated Conv.", file: "concepts/dilated.js", i18n: true },
    { id: "dwsep", area: "cnn", sec: "§18", en: "Depthwise Separable Conv.", added: true, file: "concepts/dwsep.js", i18n: true },
    { id: "resnet", area: "cnn", sec: "§21", en: "Skip Connection", added: true, file: "concepts/resnet.js", i18n: true },

    /* 序列與語言 */
    { id: "rnn", area: "seq", sec: "§22", en: "RNN & Vanishing Gradient", file: "concepts/rnn-vanishing.js", i18n: true },
    { id: "lstm", area: "seq", sec: "§22", en: "LSTM Gates", file: "concepts/lstm.js", i18n: true },
    { id: "teacher", area: "seq", sec: "§22", en: "Teacher Forcing", added: true, file: "concepts/teacher.js", i18n: true },
    { id: "seq2seq", area: "seq", sec: "§23", en: "Seq2Seq & Source-Target Attention", file: "concepts/seq2seq.js", i18n: true },
    { id: "selfatt", area: "seq", sec: "§23", en: "Self-Attention", file: "concepts/self-attention.js", i18n: true },
    { id: "mha", area: "seq", sec: "§23", en: "Multi-Head Attention", added: true, file: "concepts/mha.js", i18n: true },
    { id: "posenc", area: "seq", sec: "§23", en: "Positional Encoding", file: "concepts/posenc.js", i18n: true },
    { id: "transformer", area: "seq", sec: "§23", en: "Transformer", file: "concepts/transformer.js", i18n: true },
    { id: "w2v", area: "seq", sec: "§27", en: "Word Embedding (word2vec)", file: "concepts/w2v.js", deps: NN, i18n: true },

    /* 生成模型 */
    { id: "ae", area: "gen", sec: "§24", en: "Autoencoder", file: "concepts/ae.js", deps: NN },
    { id: "vae", area: "gen", sec: "§24", en: "Variational Autoencoder", file: "concepts/vae.js", deps: NN },
    { id: "gan", area: "gen", sec: "§30", en: "Generative Adversarial Network", file: "concepts/gan.js", deps: NN },
    { id: "diffusion", area: "gen", sec: "§30", en: "Diffusion Model", added: true, file: "concepts/diffusion.js", deps: NN },

    /* 強化學習 */
    { id: "bandit", area: "rl", sec: "§9", en: "Bandit, ε-greedy & UCB", added: true, file: "concepts/bandit.js", deps: NN },
    { id: "qlearn", area: "rl", sec: "§9", en: "Q-learning & SARSA", file: "concepts/qlearn.js", deps: NN },
  ]
);
