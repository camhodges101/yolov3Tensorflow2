import tensorflow as tf

class WeightReader:
    def __init__(self, weight_file):
        self.offset = 5
        self.all_weights = np.fromfile(weight_file, dtype='float32')
        
    def read_bytes(self, size):
        self.offset = self.offset + size
        return self.all_weights[self.offset-size:self.offset]
    
    def reset(self):
        self.offset = 5
        
        
path='yolov3.weights'


weight_reader = WeightReader(path)


def ConvertWeights():
    saver_variables=tf.train.Saver(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES))
    with tf.Session() as sess:
        variables=tf.global_variables()
        weight_reader.reset()
        for varRef in range(len(variables)):
            if variables[varRef].op.name[:5] == "ConvL":
                if variables[varRef+1].op.name[:4] == "norm":
                    print("Loading Conv and Batch Norm Weights "+str(variables[varRef].op.name))
                    filtersize=variables[varRef].shape[-1].value
                    gamma=weight_reader.read_bytes(filtersize)
                    beta=weight_reader.read_bytes(filtersize)
                    moving_mean=weight_reader.read_bytes(filtersize)
                    moving_average=weight_reader.read_bytes(filtersize)
                    convshape=(variables[varRef].shape[0].value,variables[varRef].shape[1].value,variables[varRef].shape[2].value,variables[varRef].shape[3].value)
                    convweights=weight_reader.read_bytes(np.prod(convshape))
                    variables[varRef+1].load(gamma,session=sess)
                    variables[varRef+2].load(beta,session=sess)
                    variables[varRef+3].load(moving_mean,session=sess)
                    variables[varRef+4].load(moving_average,session=sess)
                    convweights = convweights.reshape(list(reversed(convshape)))#shape=(32,3,3,3)
                    convweights = convweights.transpose([3,2,1,0])#shape=(3,3,3,32)
                    variables[varRef].load(convweights,session=sess)
                elif variables[varRef+1].op.name[:5] == "ConvB":
                    print("Loading Conv and Bias Weights "+str(variables[varRef].op.name))
                    filtersize=variables[varRef].shape[-1].value
                    biasweights=weight_reader.read_bytes(filtersize)
                    convshape=(variables[varRef].shape[0].value,variables[varRef].shape[1].value,variables[varRef].shape[2].value,variables[varRef].shape[3].value)
                    convweights=weight_reader.read_bytes(np.prod(convshape))
                    variables[varRef+1].load(biasweights,session=sess)
                    convweights = convweights.reshape(list(reversed(convshape)))#shape=(32,3,3,3)
                    convweights = convweights.transpose([3,2,1,0])#shape=(3,3,3,32)
                    variables[varRef].load(convweights,session=sess)
        save_path_W=saver_variables.save(sess, 'yolov3_TF.ckpt',write_meta_graph=False)