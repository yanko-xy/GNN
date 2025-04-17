import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Run KGAT.")
    parser.add_argument('--model_type', nargs='?', default='xx')
    parser.add_argument('--weights_path', nargs='?', default='',
                        help='Store model path.')
    parser.add_argument('--data_path', nargs='?', default='../Data/',
                        help='Input data path.')
    parser.add_argument('--proj_path', nargs='?', default='',
                        help='Project path.')

    parser.add_argument('--dataset', nargs='?', default='movie-lens',
                        help='Choose a dataset from {movie-lens, last-fm, amazon-book}')
    parser.add_argument('--pretrain', type=int, default=-1,
                        help='0: No pretrain, -1: Pretrain with the learned embeddings, 1:Pretrain with stored models.')
    parser.add_argument('--verbose', type=int, default=1,
                        help='Interval of evaluation.')
    parser.add_argument('--epoch', type=int, default=20,
                        help='Number of epoch.')

    parser.add_argument('--embed_size', type=int, default=64,
                        help='CF Embedding size.')
    parser.add_argument('--kge_size', type=int, default=64,
                        help='KG Embedding size.')
    parser.add_argument('--layer_size', nargs='?', default='[64, 32, 16]',
                        help='Output sizes of every layer')
    parser.add_argument('--sub_layer_size', nargs='?', default='[64, 32, 16]',
                        help='Output sizes of every layer')

    parser.add_argument('--batch_size', type=int, default=8192,
                        help='CF batch size.')
    parser.add_argument('--batch_size_kg', type=int, default=8192,
                        help='KG batch size.')
    parser.add_argument('--batch_size_cl', type=int, default=8192,
                    help='CL batch size.')

    parser.add_argument('--regs', nargs='?', default='[1e-5,1e-5,1e-2]',
                        help='Regularization for user and item embeddings.')
    parser.add_argument('--lr', type=float, default=0.0001,
                        help='Learning rate.')
    parser.add_argument('--kg_lr', type=float, default=0.0001,
                        help='Learning rate.')
    parser.add_argument('--cl_lr', type=float, default=0.0001,
                        help='Learning rate.')

    parser.add_argument('--drop_rate', type=float, default=0.7,
                        help='CL dropout rate.')
    parser.add_argument('--temperature', type=float, default=0.7,
                        help='Softmax temperature.')

    parser.add_argument('--adj_type', nargs='?', default='si',
                        help='Specify the type of the adjacency (laplacian) matrix from {bi, si}.')
    parser.add_argument('--adj_uni_type', nargs='?', default='sum',
                        help='Specify a loss type (uni, sum).')

    parser.add_argument('--gpu_id', type=int, default=0,
                        help='0 for NAIS_prod, 1 for NAIS_concat')

    parser.add_argument('--node_dropout', nargs='?', default='[0.1]',
                        help='Keep probability w.r.t. node dropout (i.e., 1-dropout_ratio) for each deep layer. 1: no dropout.')
    parser.add_argument('--mess_dropout', nargs='?', default='[0.1]',
                        help='Keep probability w.r.t. message dropout (i.e., 1-dropout_ratio) for each deep layer. 1: no dropout.')

    parser.add_argument('--Ks', nargs='?', default='[10, 20, 40, 60, 80, 100]',
                        help='Output sizes of every layer')

    parser.add_argument('--save_flag', type=int, default=0,
                        help='0: Disable model saver, 1: Activate model saver')

    parser.add_argument('--test_flag', nargs='?', default='part',
                        help='Specify the test type from {part, full}, indicating whether the reference is done in mini-batch')

    parser.add_argument('--report', type=int, default=0,
                        help='0: Disable performance report w.r.t. sparsity levels, 1: Show performance report w.r.t. sparsity levels')

    parser.add_argument('--use_att', type=bool, default=False,
                        help='whether using attention mechanism')
    parser.add_argument('--use_kge', type=bool, default=False,
                        help='whether using knowledge graph embedding')
    
    parser.add_argument('--l1_flag', type=bool, default=False,
                        help='Flase: using the L2 norm, True: using the L1 norm.')
    parser.add_argument('--heads', type=int, default=1)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--kg_weight_decay', type=float, default=1e-5)
    parser.add_argument('--alpha', type=float, default=0.)
    parser.add_argument('--cl_alpha', type=float, default=1.)

    return parser.parse_args()
