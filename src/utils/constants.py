from src.utils.utils import DictX

__all_model__ = DictX(
    catboost='catboost',
    lgbm='lgbm',
    xgboost='xgboost',
    mlp='mlp',
    deepstack_dae='deepstack_dae',
    bottleneck_dae='bottleneck_dae',
    transformer_dae='transformer_dae',
    tabnet='tabnet',
    tabnet_pretrainer='tabnet_pretrainer',
    dae_mlp='dae_mlp',
    gmm_dae='gmm_dae'
)

representation_key = 'representation'



JARVIS_NULL_REPLACEMENTS = {
     'BS0000133': {999999999: 0},
     'CF0000912': {-999: 0},
     'CF0100902': {-999: 0},
     'CF0100919': {-999: 0},
     'CF0100932': {-999: 0},
     'CF0300910': {-999: 0},
     'CF0300923': {-999: 0},
     'CF0300936': {-999: 0},
     'CF0600901': {-999: 0},
     'CF0600913': {-999: 0},
     'CF0600926': {-999: 0},
     'CF0600943': {-999: 0},
     'CF1200916': {-999: 0},
     'CF1200946': {-999: 0},
     'CF9900901': {-999: 0},
     'CF9900905': {-999: 0},
     'CF9900907': {-999: 0},
     'CF9900910': {-999: 0},
     'CF9900911': {-999: 0},
     'CL0000002': {999999992: 0},
     'CL0000003': {999999992: 0},
     'CL0000201': {999999900: 0},
     'CL0000202': {999999900: 0},
     'CL0000203': {999999900: 0},
     'CL0000302': {999999900: 0},
     'CL0000903': {-999: 0, 999999991: 0},
     'CL0000911': {-999: 0, 999999991: 0},
     'CL0000912': {-999: 0, 999999991: 0},
     'CL0000A01': {999999998: 0},
     'CL0100100': {999999992: 0},
     'CL0100902': {-999: 0, 999999991: 0},
     'CL0100903': {-999: 0, 999999991: 0},
     'CL0100913': {-999: 0, 999999991: 0},
     'CL0300907': {999999991: 0},
     'CL0300912': {-999: 0, 999999991: 0},
     'CL0300915': {-999: 0, 999999991: 0},
     'CL0300921': {-999: 0, 999999991: 0},
     'CL0300924': {-999: 0, 999999991: 0},
     'CL0300925': {-999: 0, 999999991: 0},
     'CL0331402': {999999900: 0},
     'CL0600912': {-999: 0, 999999991: 0},
     'CL0600923': {999999991: 0},
     'CL0600924': {-999: 0, 999999991: 0},
     'CL1200100': {999999992: 0},
     'CS0000009': {999: 0},
     'EC0001901': {999999999: 0},
     'EH0001601': {999999999: 0},
     'EH0001801': {999999999: 0},
     'EH0001901': {999999999: 0},
     'EH0002901': {999999999: 0},
     'EH0002902': {999999900: 0},
     'EH0002903': {999999900: 0},
     'EH0002904': {999999900: 0},
     'EH0002905': {999999900: 0},
     'EH0002906': {999999900: 0},
     'EH0002907': {999999900: 0},
     'EH0002908': {999999900: 0},
     'EH0002920': {999999999: 0},
     'EH0002921': {999999999: 0},
     'EH0002924': {999999999: 0},
     'EH0002933': {999999999: 0},
     'EH0011901': {999999999: 0},
     'EH0012901': {999999999: 0},
     'EH0601001': {999999999: 0},
     'EH0611903': {999999999: 0},
     'EH1201001': {999999999: 0},
     'EH1201002': {999999999: 0},
     'EH2401001': {999999999: 0},
     'EW0001601': {999999999: 0},
     'EW0002801': {999999999: 0},
     'EW0002902': {999999999: 0},
     'EW0002905': {999999999: 0},
     'EW0002906': {999999999: 0},
     'EW0003901': {999999999: 0},
     'EW0003905': {999999999: 0},
     'EW0003910': {999999999: 0},
     'EW0004901': {999999999: 0},
     'EW0004902': {999999999: 0},
     'EW0020601': {999999999: 0},
     'EW0020901': {999999999: 0},
     'EW0021902': {999999999: 0},
     'EW1201001': {999999999: 0},
     'EW2401001': {999999999: 0},
     'KC1100010': {999999900: 0},
     'KC1100016': {999999900: 0},
     'KC1100022': {999999900: 0},
     'KC6000001': {999999999: 0},
     'KC6000002': {999999999: 0},
     'KC6000003': {999999999: 0},
     'KC6000007': {999999999: 0},
     'KC6000009': {999999999: 0},
     'KC6000010': {999999999: 0},
     'KC7000001': {999999999: 0},
     'KC7000002': {999999999: 0},
     'KC7000003': {999999999: 0},
     'KC7000004': {999999999: 0},
     'KC8000001': {999999999: 0},
     'KC8000002': {999999999: 0},
     'KC8000003': {999999999: 0},
     'KC8000004': {999999999: 0},
     'KC8000005': {999999999: 0},
     'LA0000902': {-999: 0},
     'LA0000905': {-999: 0},
     'LA2400901': {999: 0},
     'LH0000194': {999: 0},
     'LH0000195': {999: 0},
     'LH0000196': {999: 0},
     'LS0000009': {-9999999: 0, 9999999: 0},
     'LS0000029': {9999995: 0},
     'LS0000034': {99: 0},
     'LS0000614': {-999: 0},
     'LU0324901': {999999900: 0},
     'LU0324902': {-99999999: 0, 99999999: 0},
     'LU0624901': {999999900: 0},
     'LU0624902': {-99999999: 0, 99999999: 0},
     'LU1224901': {-99999999: 0, 99999999: 0},
     'LU1224903': {999999900: 0},
     'PE0000047': {-99: 0},
     'PHC000023': {-999: 0},
     'RABAC0055': {-99: 0},
     'RABAC0056': {-99: 0},
     'RABAC0058': {-99: 0},
     'RABAC0078': {-99: 0},
     'RABAC0085': {-99: 0},
     'RABAC0120': {-99: 0},
     'SC0000014': {999999998: 0},
     'SC0000017': {999999998: 0},
     'SC0000019': {999999998: 0},
     'SC0000021': {999999998: 0},
     'SC0000023': {999999998: 0},
     'SC0000025': {999999998: 0},
     'SC0000030': {999999998: 0},
     'SC0000031': {999999998: 0},
     'SC0000041': {999999998: 0},
     'SC0000050': {-99: 0, 99: 0},
     'SC0000052': {999: 0},
     'SC0000060': {-99: 0, 99: 0},
     'SC0000062': {999: 0},
     'SC0000075': {999999998: 0}
}

