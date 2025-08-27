import numpy as np
try:
    import tensorflow.compat.v1 as tf  # type: ignore
except Exception as _tf_err:
    class _DummyTF:
        def __getattr__(self, item):
            raise ImportError(f"TensorFlow 未安装，无法访问 {item}: {_tf_err}")
    tf = _DummyTF()

class AISegmentationEngine:
    """第一层：AI语义分割器 (拆分自 demo_refactored_clean). 功能未改动。"""
    def __init__(self, model_path="pretrained"):
        self.model_path = model_path
        self.session = None
        self.inputs = None
        self.room_type_logit = None
        self.room_boundary_logit = None

    def load_model(self):
        print("🔧 [第1层-AI分割器] 加载DeepFloorplan模型...")
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        self.session = tf.Session(config=config)
        saver = tf.train.import_meta_graph(f"{self.model_path}/pretrained_r3d.meta")
        saver.restore(self.session, f"{self.model_path}/pretrained_r3d")
        graph = tf.get_default_graph()
        self.inputs = graph.get_tensor_by_name("inputs:0")
        self.room_type_logit = graph.get_tensor_by_name("Cast:0")
        self.room_boundary_logit = graph.get_tensor_by_name("Cast_1:0")
        print("✅ [第1层-AI分割器] 模型加载完成")

    def segment_image(self, img_array):
        print("🤖 [第1层-AI分割器] 运行神经网络推理...")
        input_batch = np.expand_dims(img_array, axis=0)
        room_type, room_boundary = self.session.run(
            [self.room_type_logit, self.room_boundary_logit],
            feed_dict={self.inputs: input_batch},
        )
        room_type = np.squeeze(room_type)
        room_boundary = np.squeeze(room_boundary)
        floorplan = room_type.copy()
        floorplan[room_boundary == 1] = 9
        floorplan[room_boundary == 2] = 10
        print("✅ [第1层-AI分割器] 神经网络推理完成")
        return floorplan
