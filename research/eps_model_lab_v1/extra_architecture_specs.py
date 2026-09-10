"""Prescore extra breadth; installed official APIs inspected, no score tuning."""
EXTRA_UNIVARIATE={
 'NLinear':{},
 'FEDformer':{'hidden_size':64,'n_head':8,'conv_hidden_size':64,'modes':4,'MovingAvg_window':5},
 'TimesNet':{'hidden_size':64,'conv_hidden_size':64,'top_k':3,'num_kernels':3},
}
EXTRA_MULTIVARIATE={
 'TimeXer':{'patch_len':4,'hidden_size':64,'n_heads':4,'e_layers':2,'d_ff':128},
 'RMoK':{'taylor_order':3,'jacobi_degree':6,'dropout':.1},
 'XLinear':{'hidden_size':64,'temporal_ff':128,'channel_ff':8},
 'StemGNN':{'n_stacks':2,'multi_layer':3,'dropout_rate':.1},
}
