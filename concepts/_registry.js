/* =========================================================
   概念清單（依 JDLA G検定 シラバス2024 第1.4版）
   - area：所屬分區 id
   - sec：シラバス節號
   - file：已完成的概念填上模組檔路徑；沒有 file ＝「準備中」
   - added：true ＝ 使用者原本清單沒有、依シラバス補充的項目
   - deps：先載入的共用檔（assets/nn.js：小型神經網路、即時訓練迴圈、折線圖；生成模型、強化學習、word2vec 用）
   新增概念的步驟：
     1. 寫 concepts/<id>.js（呼叫 Atlas.define("<id>", {...})）
     2. 在下面對應項目加上 file: "concepts/<id>.js"
   ========================================================= */
const NN = ["assets/nn.js"];
Atlas.registry(
  [
    { id: "ml",    name: "機械學習基礎",       sec: "§7–10" },
    { id: "learn", name: "神經網路怎麼學習",   sec: "§12–16・19" },
    { id: "cnn",   name: "卷積與影像",         sec: "§18・20・21" },
    { id: "seq",   name: "序列與語言",         sec: "§22・23・27" },
    { id: "gen",   name: "生成模型",           sec: "§24・30" },
    { id: "rl",    name: "強化學習",           sec: "§9・29" },
  ],
  [
    /* 機械學習基礎 */
    { id: "kernel",   area: "ml", sec: "§7",  jp: "SVM とカーネルトリック", zh: "最大間隔與把資料升維後分開", en: "SVM & Kernel Trick", added: true, file: "concepts/svm.js" },
    { id: "kmeans",   area: "ml", sec: "§8",  jp: "k-means 法",             zh: "反覆「分組→移動中心」的分群", en: "k-means Clustering", added: true, file: "concepts/kmeans.js" },
    { id: "pca",      area: "ml", sec: "§8",  jp: "主成分分析（PCA）",      zh: "找出資料散得最開的方向來降維", en: "Principal Component Analysis", added: true, file: "concepts/pca.js" },
    { id: "roc",      area: "ml", sec: "§10", jp: "ROC 曲線と AUC",         zh: "移動判定門檻時，命中與誤報怎麼一起變", en: "ROC Curve & AUC", added: true, file: "concepts/roc.js" },

    /* 神經網路怎麼學習 */
    { id: "gd",       area: "learn", sec: "§16", jp: "勾配降下法と学習率", zh: "沿斜坡往下走，步伐大小決定成敗", en: "Gradient Descent & Learning Rate", file: "concepts/gradient-descent.js" },
    { id: "sgd",      area: "learn", sec: "§16", jp: "確率的勾配降下法とミニバッチ", zh: "每次只看一部分資料來估斜率", en: "SGD & Mini-batch", added: true, file: "concepts/sgd.js" },
    { id: "optim",    area: "learn", sec: "§16", jp: "モーメンタムと Adam", zh: "加上慣性與自動調步伐的最佳化方法", en: "Momentum & Adam", added: true, file: "concepts/optim.js" },
    { id: "backprop", area: "learn", sec: "§15", jp: "誤差逆伝播法と連鎖律", zh: "把誤差從輸出一層層往回分攤", en: "Backpropagation & Chain Rule", file: "concepts/backprop.js" },
    { id: "vanish",   area: "learn", sec: "§12・15", jp: "勾配消失問題", zh: "層數一深，前面的層幾乎學不動", en: "Vanishing Gradient", file: "concepts/vanishing.js" },
    { id: "act",      area: "learn", sec: "§12", jp: "活性化関数", zh: "Sigmoid、tanh、ReLU 的形狀與斜率", en: "Activation Functions", file: "concepts/activation.js" },
    { id: "loss",     area: "learn", sec: "§13", jp: "誤差関数", zh: "怎麼把「錯多少」變成一個數字", en: "Loss Functions", added: true, file: "concepts/loss.js" },
    { id: "overfit",  area: "learn", sec: "§10・14", jp: "過学習と正則化（L1・L2）", zh: "背答案的模型，和替它加上的約束", en: "Overfitting & Regularization", file: "concepts/overfitting.js" },
    { id: "dropout",  area: "learn", sec: "§14", jp: "ドロップアウト", zh: "訓練時隨機關掉一部分神經元", en: "Dropout", file: "concepts/dropout.js" },
    { id: "norm",     area: "learn", sec: "§19", jp: "正規化層", zh: "沿哪個方向把數值拉回同一尺度", en: "Normalization Layers", file: "concepts/norm.js" },

    /* 卷積與影像 */
    { id: "conv",     area: "cnn", sec: "§18", jp: "畳み込み（CNN）", zh: "濾波器滑過圖像，算出特徵圖", en: "Convolution", file: "concepts/convolution.js" },
    { id: "pool",     area: "cnn", sec: "§20", jp: "プーリング", zh: "縮小特徵圖，只留下重點", en: "Pooling", file: "concepts/pooling.js" },
    { id: "dilated",  area: "cnn", sec: "§18", jp: "受容野と Dilated Convolution", zh: "一個輸出看得到輸入的多大範圍", en: "Receptive Field & Dilated Conv.", file: "concepts/dilated.js" },
    { id: "dwsep",    area: "cnn", sec: "§18", jp: "Depthwise Separable Convolution", zh: "把卷積拆成兩步，參數大減", en: "Depthwise Separable Conv.", added: true, file: "concepts/dwsep.js" },
    { id: "resnet",   area: "cnn", sec: "§21", jp: "スキップ結合（ResNet）", zh: "讓訊號和梯度走捷徑", en: "Skip Connection", added: true, file: "concepts/resnet.js" },

    /* 序列與語言 */
    { id: "rnn",      area: "seq", sec: "§22", jp: "RNN と勾配消失", zh: "沿時間展開，看狀態往後傳、梯度往回淡", en: "RNN & Vanishing Gradient", file: "concepts/rnn-vanishing.js" },
    { id: "lstm",     area: "seq", sec: "§22", jp: "LSTM のゲート", zh: "忘記閘、輸入閘、輸出閘各管什麼", en: "LSTM Gates", file: "concepts/lstm.js" },
    { id: "teacher",  area: "seq", sec: "§22", jp: "教師強制", zh: "訓練時餵正解，而不是自己的預測", en: "Teacher Forcing", added: true, file: "concepts/teacher.js" },
    { id: "seq2seq",  area: "seq", sec: "§23", jp: "Seq2Seq と Source-Target Attention", zh: "翻譯時回頭看原文的哪個詞", en: "Seq2Seq & Source-Target Attention", file: "concepts/seq2seq.js" },
    { id: "selfatt",  area: "seq", sec: "§23", jp: "Self-Attention", zh: "句中每個詞決定要關注誰", en: "Self-Attention", file: "concepts/self-attention.js" },
    { id: "mha",      area: "seq", sec: "§23", jp: "Multi-Head Attention", zh: "同時用好幾組視角看關係", en: "Multi-Head Attention", added: true, file: "concepts/mha.js" },
    { id: "posenc",   area: "seq", sec: "§23", jp: "位置エンコーディング", zh: "告訴模型每個詞排在第幾個", en: "Positional Encoding", file: "concepts/posenc.js" },
    { id: "transformer", area: "seq", sec: "§23", jp: "Transformer", zh: "把注意力零件接起來的整體架構", en: "Transformer", file: "concepts/transformer.js" },
    { id: "w2v",      area: "seq", sec: "§27", jp: "単語埋め込み（word2vec）", zh: "從上下文學出詞的向量", en: "Word Embedding (word2vec)", file: "concepts/w2v.js", deps: NN },

    /* 生成模型 */
    { id: "ae",       area: "gen", sec: "§24", jp: "オートエンコーダ", zh: "壓縮再還原，學出精簡表示", en: "Autoencoder", file: "concepts/ae.js", deps: NN },
    { id: "vae",      area: "gen", sec: "§24", jp: "変分オートエンコーダ（VAE）", zh: "把壓縮結果變成一個分布", en: "Variational Autoencoder", file: "concepts/vae.js", deps: NN },
    { id: "gan",      area: "gen", sec: "§30", jp: "敵対的生成ネットワーク（GAN）", zh: "生成器與識別器的互相較勁", en: "Generative Adversarial Network", file: "concepts/gan.js", deps: NN },
    { id: "diffusion", area: "gen", sec: "§30", jp: "Diffusion Model", zh: "一步步加雜訊，再學會一步步去除", en: "Diffusion Model", added: true, file: "concepts/diffusion.js", deps: NN },

    /* 強化學習 */
    { id: "bandit",   area: "rl", sec: "§9", jp: "バンディットと ε-greedy 方策", zh: "探索與利用的取捨（含 UCB 方策）", en: "Bandit, ε-greedy & UCB", added: true, file: "concepts/bandit.js", deps: NN },
    { id: "qlearn",   area: "rl", sec: "§9", jp: "Q 学習と SARSA", zh: "懸崖邊：最短的路，還是安全的路", en: "Q-learning & SARSA", file: "concepts/qlearn.js", deps: NN },
  ]
);
