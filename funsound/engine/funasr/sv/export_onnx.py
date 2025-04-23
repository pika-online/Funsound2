from speakerlab.bin.infer_sv import *


def export( cache_dir,model_id, device='cpu'):

    conf = supports[model_id]
    cache_dir = snapshot_download(
        model_id=model_id,
        revision=conf['revision'],
        cache_dir=cache_dir
    )
    cache_dir = pathlib.Path(cache_dir)
    target_onnx_file = f'{cache_dir}/model.onnx'

    pretrained_model = cache_dir / conf['model_pt']
    pretrained_state = torch.load(pretrained_model, map_location='cpu')

    model = conf['model']
    embedding_model = dynamic_import(model['obj'])(**model['args'])
    embedding_model.load_state_dict(pretrained_state)
    embedding_model.to(device)
    embedding_model.eval()


    dummy_input = torch.randn(1, 345, 80)
    torch.onnx.export(embedding_model,
                      dummy_input,
                      target_onnx_file,
                      export_params=True,
                      opset_version=11,
                      do_constant_folding=True,
                      input_names=['feature'],
                      output_names=['embedding'],
                      dynamic_axes={'feature': {0: 'batch_size', 1: 'frame_num'},
                                    'embedding': {0: 'batch_size'}})


if __name__ == "__main__":

    cache_dir = 'models'
    model_id = "iic/speech_eres2net_base_200k_sv_zh-cn_16k-common"
    export(cache_dir,model_id)

