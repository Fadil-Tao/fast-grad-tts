# %% [cell 4] Experiment config — override params.py values in-process
# These attributes are read by the training cell at runtime, so setting them here
# is enough (no need to edit params.py on disk).
import params

params.log_dir = LOG_DIR        # checkpoints -> Drive (survive Colab disconnects)
params.batch_size = 16          # lower (e.g. 8) if you hit CUDA OOM on Colab
params.save_every = 1           # save a checkpoint every N epochs
params.n_spks = 1               # LJSpeech is single-speaker
params.pe_scale = 1000          # positional-encoding scale (matches released grad-tts.pt)

print('batch_size =', params.batch_size,
      '| out_size =', params.out_size,
      '| log_dir =', params.log_dir)
