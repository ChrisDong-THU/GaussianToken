CUDA_VISIBLE_DEVICES="0" python evaluate.py \
--config_file ./logs/gqgan/test1-adam-1e4w0/config.yaml \
--ckpt_path ./logs/gqgan/test1-adam-1e4w0/checkpoints/last.ckpt \
--batch_size 64