## Transcription Training
```mermaid
graph LR
    %% Inputs
    AT_pair[Audio-Text pair] --> A_in[Audio]
    AT_pair --> T_in[Text]
    A_in --> A_emb[Audio Embedding]
    T_in --> T_emb[Text Embedding]

    %% Embeddings to Encoder
    A_emb --> Enc[Shared Encoder]
    T_emb --> Enc

    %% Encoder to Latents
    Enc --> ZA[z-audio]
    Enc --> ZT[z-text]

    %% Audio Transcription Branch
    ZA --> Tr_dec[Transcription Head]
    Tr_dec --> A_loss[Audio Transcription Loss]

    %%  Contrastive Alignment Branch
    ZA --> Proj[Audio Projector]
    Proj --> C_loss[SigLip Loss]
    ZT --> C_loss
```

## Audio-Text Tool call training V1
```mermaid
graph LR
    %% Inputs
    AT_pair[Audio-Text prompt pair] --> A_in[Audio Prompt]
    AT_pair --> T_in[Text Prompt]
    A_in --> A_emb[Audio Embedding]
    T_in --> T_emb[Text Embedding]
    P_in[Tool List] --> T_emb

    %% Embeddings to Encoder
    T_emb --> Enc[Shared Encoder]
    A_emb --> Enc

    %% Encoder to Latents
    Enc --> ZA[z-audio]
    Enc --> ZP[z-list]
    Enc --> ZT[z-text]

    %%  Contrastive Alignment Branch
    ZA --> Proj[Audio Projector]
    Proj --> C_loss[SigLip Loss]
    ZT --> C_loss

    %% Merged Latents
    ZA --> ZAP[z-audio-list]
    ZP --> ZTP
    ZP --> ZAP
    ZT --> ZTP[z-text-list]

    %% Tool Call Branch
    ZAP --> Tool_dec[Tool Call Head]
    ZTP --> Tool_dec
    Tool_dec --> Tool_loss[Text Tool Call Loss]
    Tool_dec --> ATool_loss[Audio Tool Call Loss]
```

## Audio-Text Tool call training V2
```mermaid
graph LR
    %% Inputs
    AT_pair[Audio-Text prompt pair] --> A_in[Audio Prompt]
    AT_pair --> T_in[Text Prompt]
    A_in --> A_emb[Audio Embedding]
    T_in --> T_emb[Text Embedding]

    %% Embeddings to Encoder
    T_emb --> Enc[Shared Encoder]
    A_emb --> Enc

    %% Encoder to Latents
    Enc --> ZA[z-audio]
    Enc --> ZT[z-text]

    %%  Contrastive Alignment Branch
    ZA --> Proj[Audio Projector]
    Proj --> C_loss[SigLip Loss]
    ZT --> C_loss

    %% Tool Call Branch
    ZA --> Tool_dec[Tool Call Head]
    ZT --> Tool_dec
    P_in[Tool List] --> Tool_dec
    Tool_dec --> Tool_loss[Text Tool Call Loss]
    Tool_dec --> ATool_loss[Audio Tool Call Loss]
```

## Audio-Text Tool call training V3
```mermaid
graph LR
    %% Inputs
    AT_pair[Audio-Text prompt pair] --> A_in[Audio Prompt]
    AT_pair --> T_in[Text Prompt]
    A_in --> A_emb[Audio Embedding]
    T_in --> T_emb[Text Embedding]
    A_emb --> A_emb_f[Audio + '$' + Tool List Embedding]
    T_emb --> T_emb_f[Text + '$' + Tool List Embedding]
    Tool_emb[Tool List Embedding] --> A_emb_f
    Tool_emb --> T_emb_f

    %% Embeddings to Encoder
    T_emb_f --> Enc[Shared Encoder]
    A_emb_f --> Enc

    %% Encoder to Latents
    Enc --> ZA[z-audio]
    Enc --> ZT[z-text]

    %%  Contrastive Alignment Branch
    ZA --> Proj[Audio Projector]
    Proj --> C_loss[SigLip Loss]
    ZT --> C_loss

    %% Tool Call Branch
    ZA --> Tool_dec[Tool Call Head]
    ZT --> Tool_dec
    Tool_dec --> Tool_loss[Text Tool Call Loss]
    Tool_dec --> ATool_loss[Audio Tool Call Loss]
```

Audio transcription training and tool call training can be done at the same time, while the Audio tool call training should be done separately at the end.