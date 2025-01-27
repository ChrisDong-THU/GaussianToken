CUDA_VISIBLE_DEVICES="1" python reconstruct.py \
--config_file ./logs/gqgan/cifar-gqgan-gs64-cb1024/config.yaml \
--ckpt_path ./logs/gqgan/cifar-gqgan-gs64-cb1024/checkpoints/last.ckpt \
--batch_size 64