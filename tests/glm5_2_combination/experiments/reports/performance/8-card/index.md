# 8-card eager/Inductor performance

| topology | eager step | Inductor step | speedup | analysis |
|---|---:|---:|---:|---|
| cp8 | 2.880988 s | 2.813859 s | 1.0239x | [cp8](cp8/analysis.md) |
| ddp8 | 0.420374 s | 0.427276 s | 0.9838x | [ddp8](ddp8/analysis.md) |
| ep8 | 0.791739 s | 0.777959 s | 1.0177x | [ep8](ep8/analysis.md) |
| fsdp2-cp4 | 1.477067 s | 1.422781 s | 1.0382x | [fsdp2-cp4](fsdp2-cp4/analysis.md) |
| fsdp2-pp4 | 3.488729 s | 3.310308 s | 1.0539x | [fsdp2-pp4](fsdp2-pp4/analysis.md) |
| fsdp2-tp2-pp2 | 14.010456 s | 9.097924 s | 1.5400x | [fsdp2-tp2-pp2](fsdp2-tp2-pp2/analysis.md) |
| fsdp2-tp4 | 3.333390 s | 2.174089 s | 1.5332x | [fsdp2-tp4](fsdp2-tp4/analysis.md) |
| fsdp2-tp4-ep8 | 3.768185 s | 2.589503 s | 1.4552x | [fsdp2-tp4-ep8](fsdp2-tp4-ep8/analysis.md) |
| fsdp4-tp2 | 1.628292 s | 1.083835 s | 1.5023x | [fsdp4-tp2](fsdp4-tp2/analysis.md) |
| fsdp8 | 0.425111 s | 0.417484 s | 1.0183x | [fsdp8](fsdp8/analysis.md) |
| pp8 | 4.185296 s | 3.849400 s | 1.0873x | [pp8](pp8/analysis.md) |
| tp2-cp4 | 6.516221 s | 4.359172 s | 1.4948x | [tp2-cp4](tp2-cp4/analysis.md) |
| tp8 | 6.424063 s | 4.240238 s | 1.5150x | [tp8](tp8/analysis.md) |
