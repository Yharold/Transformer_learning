from transformer import *

num_hiddens, num_layers, dropout, batch_size, num_steps = 32, 2, 0.1, 64, 10
lr, num_epochs, device = 0.005, 200, try_gpu()
ffn_num_input, ffn_num_hiddens, num_heads = 32, 64, 4
key_size, query_size, value_size = 32, 32, 32
norm_shape = [32]
train_iter, src_vocab, tgt_vocab = load_data_nmt(batch_size, num_steps)
encoder = TransformerEncoder(
    len(src_vocab),
    key_size,
    query_size,
    value_size,
    num_hiddens,
    norm_shape,
    ffn_num_input,
    ffn_num_hiddens,
    num_heads,
    num_layers,
    dropout,
)
decoder = TransformerDecoder(
    len(tgt_vocab),
    key_size,
    query_size,
    value_size,
    num_hiddens,
    norm_shape,
    ffn_num_input,
    ffn_num_hiddens,
    num_heads,
    num_layers,
    dropout,
)
net = EncoderDecoder(encoder, decoder)
train_seq2seq(net, train_iter, lr, num_epochs, tgt_vocab, device)

# if "net.pt" not in os.listdir():
#     torch.save(net.state_dict(), "net.pt")
# else:
#     net.load_state_dict(torch.load("net.pt"))

engs = ["go .", "i lost .", "he's calm .", "i'm home ."]
fras = ["va !", "j'ai perdu .", "il est calme .", "je suis chez moi ."]
for eng, fra in zip(engs, fras):
    translation, dec_attention_weight_seq = predict_seq2seq(
        net, eng, src_vocab, tgt_vocab, num_steps, device, True
    )
    print(f"{eng} => {translation}, ", f"bleu {bleu(translation, fra, k=2):.3f}")
