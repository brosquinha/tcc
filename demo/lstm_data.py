tests_data = {
    'CNN': {
        'TEC': {
            'Raiva': [[3527, 377], [191, 116]],
            'Medo': [[3287, 363], [244, 317]],
            'Alegria': [[1982, 573], [519, 1137]],
            'Tristeza': [[2811, 658], [353, 389]],
        },
        'SemEval': {
            'Raiva': [[1135, 288], [252, 522]],
            'Medo': [[1623, 212], [180, 182]],
            'Alegria':  [[1041, 303], [242, 611]],
            'Tristeza': [[1176, 371], [288, 362]]
        }
    },
    'LSTM bidirecional': {
        'TEC': {
            'Raiva': [[3405, 490], [170, 146]],
            'Medo': [[3145, 524], [177, 365]],
            'Alegria': [[1932, 678], [402, 1199]],
            'Tristeza': [[2794, 611], [418, 388]],
        },
        'SemEval': {
            'Raiva': [[1100, 304], [185, 608]],
            'Medo': [[1363, 448], [83, 303]],
            'Alegria': [[1071, 300], [157, 669]],
            'Tristeza': [[1213, 316], [282, 386]],
        },
    },
    'Concatenação LSTM-bi e CNN': {
        'TEC': {
            'Raiva': [[3420, 475], [165, 151]],
            'Medo': [[3157, 494], [218, 342]],
            'Alegria': [[1978, 612], [424, 1197]],
            'Tristeza': [[2843, 615], [340, 413]],
        },
        'SemEval': {
            'Raiva': [[1192, 195], [269, 541]],
            'Medo': [[1715, 113], [180, 189]],
            'Alegria': [[1045, 288], [185, 679]],
            'Tristeza': [[1205, 349], [252, 391]],
        },
    },
    'LSTM-bi seguida de CNN': {
        'TEC': {
            'Raiva': [[3506, 407], [163, 135]],
            'Medo': [[3288, 348], [231, 344]],
            'Alegria': [[1846, 718], [393, 1254]],
            'Tristeza': [[2569, 830], [265, 547]],
        },
        'SemEval': {
            'Raiva': [[1101, 283], [188, 625]],
            'Medo': [[1619, 206], [124, 248]],
            'Alegria': [[1112, 243], [204, 638]],
            'Tristeza': [[1208, 349], [240, 400]],
        },
    },
    'Função de custo Dice': {
        'TEC': {
            'Raiva': [[3413, 477], [160, 161]],
            'Medo': [[3202, 407], [209, 393]],
            'Alegria': [[1894, 649], [431, 1237]],
            'Tristeza': [[2694, 733], [291, 493]],
        },
        'SemEval': {
            'Raiva': [[1169, 266], [233, 529]],
            'Medo': [[1499, 318], [102, 278]],
            'Alegria': [[621, 718], [53, 805]],
            'Tristeza': [[1194, 357], [237, 409]],
        },
    },
    'Duas camadas LSTM e CNN com max pooling local': {
        'TEC': {
            'Raiva': [[3531, 378], [158, 144]],
            'Medo': [[3279, 382], [205, 345]],
            'Alegria': [[2013, 539], [503, 1156]],
            'Tristeza': [[2781, 636], [370, 424]],
        },
        'SemEval': {
            'Raiva': [[1185, 219], [232, 561]],
            'Medo': [[1627, 190], [132, 248]],
            'Alegria': [[1148, 199], [223, 627]],
            'Tristeza': [[1120, 441], [179, 457]],
        },
    },
    'Duas camadas LSTM e CNN com max pooling único': {
        'TEC': {
            'Raiva': [[3508, 410], [155, 138]],
            'Medo': [[3243, 401], [212, 355]],
            'Alegria': [[1997, 582], [455, 1177]],
            'Tristeza': [[2867, 563], [365, 416]],
        },
        'SemEval': {
            'Raiva': [[1182, 232], [220, 563]],
            'Medo': [[1653, 158], [141, 245]],
            'Alegria': [[1070, 265], [163, 699]],
            'Tristeza': [[1195, 358], [214, 430]],
        },
    }
}

arq_data = {
    'CNN': 'cnn_only.png',
    'LSTM bidirecional': 'lstm_only.png',
    'Concatenação LSTM-bi e CNN': 'lstm_cnn_conc.png',
    'LSTM-bi seguida de CNN': 'lstm_and_cnn.png',
    'Função de custo Dice': 'lstm_and_cnn.png',
    'Duas camadas LSTM e CNN com max pooling local': 'lstm_cnn_two_maxpool.png',
    'Duas camadas LSTM e CNN com max pooling único': 'lstm_cnn_one_maxpool.png'
}

confiability_data = {
    'SemEval': {
        'Raiva': [[13381, 6115], [1009, 546]],
        'Medo': [[15342, 2893], [2261, 555]],
        'Alegria': [[7457, 5354], [4522, 3718]],
        'Tristeza': [[11869, 5352], [2640, 1190]],
    },
    'TEC': {
        'Raiva': [[6022, 1001], [3489, 471]],
        'Medo': [[8096, 1039], [1683, 165]],
        'Alegria': [[4100, 2564], [2723, 1596]],
        'Tristeza': [[6227, 1523], [2522, 711]],
    }
}
