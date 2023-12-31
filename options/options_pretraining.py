# Sequence based model arguments
encoder_arguments = {
   "t_input_size": 150,
   "s_input_size": 24,
    # "t_input_size": 12288,
    # "s_input_size": 12288,
    "kernel_size": 5,
    "stride": 1,
    "padding": 2,
    "factor": 2,
    "hidden_size": 512,
    "num_head": 4,
    "num_layer": 1,
    "granularity": 4,
    "encoder": "Transformer",
    "num_class": 128
 }

data_path = "../HiCo-data"
# data_path = "/root/autodl-tmp/HiCo-data"

class opts_pointcloud():
    def __init__(self):
        self.encoder_args = encoder_arguments

        # feeder
        self.train_feeder_args = {
            "data_path": data_path + "/pr_dataset_pointcloud/train_data_point.npy",
            "num_frame_path": data_path + "/pr_dataset_pointcloud/train_num_frame.npy",
            "l_ratio": [0.1, 1],
            "input_size": 32
        }

class opts_ntu_60_cross_view():
  def __init__(self):

   self.encoder_args = encoder_arguments

   # feeder
   self.train_feeder_args = {
     "data_path": data_path + "/NTU-RGB-D-60-AGCN/xview/train_data_joint.npy",
     "num_frame_path": data_path + "/NTU-RGB-D-60-AGCN/xview/train_num_frame.npy",
     "l_ratio": [0.1,1],
     "input_size": 64
   }


class opts_ntu_60_cross_subject():  # default
    def __init__(self):
        self.encoder_args = encoder_arguments
   
        # feeder
        self.train_feeder_args = {
            "data_path": data_path + "/NTU-RGB-D-60-AGCN/xsub/train_data_joint.npy",
            "num_frame_path": data_path + "/NTU-RGB-D-60-AGCN/xsub/train_num_frame.npy",
            "l_ratio": [0.1, 1],
            "input_size": 8
   }


class opts_ntu_120_cross_subject():
  def __init__(self):

   self.encoder_args = encoder_arguments
   
   # feeder
   self.train_feeder_args = {
     "data_path": data_path + "/NTU-RGB-D-120-AGCN/xsub/train_data_joint.npy",
     "num_frame_path": data_path + "/NTU-RGB-D-120-AGCN/xsub/train_num_frame.npy",
     "l_ratio": [0.1,1],
     "input_size": 64
   }


class opts_ntu_120_cross_setup():

  def __init__(self):

   self.encoder_args = encoder_arguments
   
   # feeder
   self.train_feeder_args = {
     "data_path": data_path + "/NTU-RGB-D-120-AGCN/xsetup/train_data_joint.npy",
     "num_frame_path": data_path + "/NTU-RGB-D-120-AGCN/xsetup/train_num_frame.npy",
     "l_ratio": [0.1,1],
     "input_size": 64
   }


class  opts_pku_part1_cross_subject():

  def __init__(self):

   self.encoder_args = encoder_arguments
   
   # feeder
   self.train_feeder_args = {
     "data_path": data_path + "/pku_part1_frame50/xsub/train_position.npy",
     "num_frame_path": None ,
     "l_ratio": [0.1,1],
     "input_size": 64
   }


class  opts_pku_part2_cross_subject():

  def __init__(self):

   self.encoder_args = encoder_arguments
   
   # feeder
   self.train_feeder_args = {
     "data_path": data_path + "/pku_part2_frame50/xsub/train_position.npy",
     "num_frame_path": None ,
     "l_ratio": [0.1,1],
     "input_size": 64
   }

