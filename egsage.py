import paddle
import paddle.fluid as fluid
from paddle.fluid.dygraph import Layer
from functools import partial
class egsage(Layer):
    def __init__(self, in_channels, out_channels, edge_channels):
        super(egsage, self).__init__()
        self.in_channels=in_channels
        self.out_channels=out_channels
        self.edge_channels=edge_channels

        self.message_lin=partial(paddle.fluid.layers.fc, size=out_channels, act="relu")
        self.agg_lin=partial(paddle.fluid.layers.fc, size=out_channels, act="relu")

    def forward(self, x, edge_attr, edge_index):
        #message
        #print(edge_index.shape, x.shape)
        m_j=fluid.layers.gather(x, edge_index[1])#[edge_index[1]]
        #print(m_j.shape)
        m_j=fluid.layers.reshape(paddle.fluid.layers.concat(input=[m_j, edge_attr], axis=-1), [-1, 1024+1])
        print(m_j.shape)
        m_j=self.message_lin(m_j)
        #aggr
        aggr_out=fluid.layers.scatter(x*0.0, edge_index[0], m_j)
        aggr_size=fluid.layers.scatter(x*0.0, edge_index[0], m_j*0.0+1.0)
        aggr_out=aggr_out/(aggr_size+1e-9)
        #update
        aggr_out=fluid.layers.concat(input=[x, aggr_out], axis=1)
        aggr_out=self.agg_lin(aggr_out)
        return aggr_out
