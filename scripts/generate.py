from absl import app, flags, logging
import pdb
import torch, torchaudio, argparse, os, tqdm, re, gin
import cached_conv as cc

import numpy as np

try:
    import rave
except:
    import sys, os 
    sys.path.append(os.path.abspath('.'))
    import rave


FLAGS = flags.FLAGS
flags.DEFINE_string('model', required=True, default=None, help="model path")
flags.DEFINE_multi_string('input', required=True, default=None, help="model inputs (file or folder)")
flags.DEFINE_string('out_path', 'generations', help="output path")
flags.DEFINE_string('name', None, help="name of the model")
flags.DEFINE_integer('gpu', default=-1, help='GPU to use')
flags.DEFINE_bool('stream', default=False, help='simulates streaming mode')
flags.DEFINE_integer('chunk_size', default=None, help="chunk size for encoding/decoding (default: full file)")


def get_audio_files(path):
    audio_files = []
    valid_exts = rave.core.get_valid_extensions()
    for root, _, files in os.walk(path):
        valid_files = list(filter(lambda x: os.path.splitext(x)[1] in valid_exts, files))
        audio_files.extend([(path, os.path.join(root, f)) for f in valid_files])
    return audio_files

def write_latents(data_tensor, fname):
    """
    Convert tensor from [1, 4, 130] to [130, 4] and save as binary
    """
    # This reshapes from [1, 4, 130] to [130, 4]
    data_reshaped = data_tensor.squeeze(0).transpose(0, 1)  # now [130, 4]

    # Convert to numpy and ensure float32 for Max/MSP compatibility
    data_np = data_reshaped.numpy().astype(np.float32)

    print(f"Original shape: {data_tensor.shape}")
    print(f"Reshaped for Max: {data_np.shape}")  # Should be [130, 4]
    print(f"First few samples:\n{data_np[:5]}")

    # Save as binary file for Max/MSP
    data_np.tofile(fname+'_4d.raw')



def main(argv):
    torch.set_float32_matmul_precision('high')
    cc.use_cached_conv(FLAGS.stream)

    model_path = FLAGS.model
    paths = FLAGS.input
    # load model
    logging.info("_building rave_")
    is_scripted = False
    if not os.path.exists(model_path):
        logging.error('path %s does not seem to exist.'%model_path)
        exit()
    if os.path.splitext(model_path)[1] == ".ts":
        model = torch.jit.load(model_path)
        is_scripted = True
    else:
        config_path = rave.core.search_for_config(model_path)
        if config_path is None:
            logging.error('config not found in folder %s'%model_path)
        gin.parse_config_file(config_path)
        model = rave.RAVE()
        run = rave.core.search_for_run(model_path)
        if run is None:
            logging.error("run not found in folder %s"%model_path)
        model = model.load_from_checkpoint(run)

    # device
    if FLAGS.gpu >= 0:
        device = torch.device('cuda:%d'%FLAGS.gpu)
        model = model.to(device)
    else:
        device = torch.device('cpu')

    # make output directories
    if FLAGS.name is None:
        FLAGS.name = "_".join(os.path.basename(model_path).split('_')[:-1])
    out_path = os.path.join(FLAGS.out_path, FLAGS.name)
    os.makedirs(out_path, exist_ok=True)

    # parse inputs
    audio_files = sum([get_audio_files(f) for f in paths], [])
    logging.info("_building rave_   4 get_minimum_size")
    receptive_field = rave.core.get_minimum_size(model)
    logging.info("_building rave_   5 get_minimum_size")

    progress_bar = tqdm.tqdm(audio_files)
    cc.MAX_BATCH_SIZE = 8


    for i, (d, f) in enumerate(progress_bar):
        #TODO reset cache
            
        try:
            x, sr = torchaudio.load(f)
        except: 
            logging.warning('could not open file %s.'%f)
            continue
        progress_bar.set_description(f)

        # load file
        if sr != model.sr:
            x = torchaudio.functional.resample(x, sr, model.sr)
        if model.n_channels != x.shape[0]:
            if model.n_channels < x.shape[0]:
                x = x[:model.n_channels]
            else:
                print('[Warning] file %s has %d channels, but model has %d channels ; skipping'%(f, model.n_channels))
        x = x.to(device)
        if FLAGS.stream:
            if FLAGS.chunk_size:
                assert FLAGS.chunk_size > receptive_field, "chunk_size must be higher than models' receptive field (here : %s)"%receptive_field
                x = list(x.split(FLAGS.chunk_size, dim=-1))
                if x[-1].shape[0] < FLAGS.chunk_size:
                    x[-1] = torch.nn.functional.pad(x[-1], (0, FLAGS.chunk_size - x[-1].shape[-1]))
                x = torch.stack(x, 0)
            else:
                x = x[None]
            
            # forward into model
            print(f' NOW Forward to model ...............')
            out = []
            for x_chunk in x:
                x_chunk = x_chunk.to(device)
                out_tmp = model(x_chunk[None])
                out.append(out_tmp)
            out = torch.cat(out, -1)
        else:
            
            if 1 :
                z = model.encode(x[None])
                #if os.path.splitext(model_path)[1] != ".ts":
                #    z = model.encoder.reparametrize(z)[0]

                ## WRITE OUTPUT LATENTS HERE
                print(f"encode latents z.shape = {z.shape}")
                print("encoded latents z stats:", z.mean().item(), z.std().item())

                out = model.decode(z)
            else :
                out = model.forward(x[None])

        # save file
        cleaned_f = re.sub(d, "", f)  # remove unwanted pattern from f
        snd_out_path = os.path.join(out_path, cleaned_f.lstrip("/"))
        print(f"Saving output to {snd_out_path}")
        torchaudio.save(snd_out_path, out[0].cpu(), sample_rate=model.sr)

        # and write the latents
        write_latents(z, os.path.splitext(snd_out_path)[0]) # strip the .wav file extension

if __name__ == "__main__": 
    app.run(main)