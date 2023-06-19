from workflow.scripts.generate_splits import generate_random_split, generate_matched_split, write_splitfile
import os
import json
import numpy as np

# Create temporary files for testing
features_path = "tests/data/generate_splits-features.npy"
targets_path = "tests/data/generate_splits-targets.npy"
matching_path = "tests/data/generate_splits-matching.npy"
split_path = "tests/data/generate_splits-split.npy"

# Generate sample files
features = np.array([[0.0607495227, 0.317397108, 0.396117838, 0.352398709, 0.370560179, -0.276183696, -0.212327711, -0.0237740466, -0.299279555, 0.298851471], [-0.311822933, -0.431645885, 0.388923484, 0.386021866, -0.469723844, -0.378089017, 0.0680322522, 0.0917847357, -0.321037578, -0.0988474115], [-0.182432419, -0.392936576, -0.498293632, 0.270046835, -0.0171265311, 0.498123994, -0.094498768, 0.0941119303, -0.0105254538, 0.504282091], [0.103601936, 0.251767844, -0.345475262, 0.316369629, 0.21767082, 0.363753819, 0.38050316, -0.0559233605, 0.386685094, -0.026469004], [-0.295405356, 0.177332219, 0.000339197894, -0.259671463, -0.205334392, 0.0768400645, 0.254258948, 0.42284867, -0.410427993, -0.25702264], [-0.210620572, -0.0218800413, 0.0300683178, -0.376087421, -0.324471845, -0.183666736, 0.0647103182, -0.215033222, -0.126601721, 0.161359643], [-0.485687747, -0.0925150977, 0.185041668, 0.139315276, -0.341536566, 0.337182434, -0.017309071, -0.215437234, 0.549335953, 0.409269626], [-0.249164162, -0.25047505, 0.283612462, 0.168928255, 0.0417997842, -0.270878138, -0.0487344642, 0.418221568, 0.161212537, 0.0770187792], [0.268558391, -0.104372972, 0.29191931, -0.260242695, -0.252616658, -0.30841946, -0.0848639287, 0.388898066, 0.255569844, -0.504933978], [0.210625626, -0.130255237, -0.562228504, 0.468621358, -0.153782153, -0.109239831, -0.115259743, 0.463530316, -0.269878366, -0.265067342], [0.243288014, 0.415171391, 0.292885842, -0.465471032, 0.464009198, 0.0607988905, 0.344699895, 0.231197693, -0.190563276, -0.16041712], [-0.350248382, -0.0765060248, -0.389847093, -0.169333014, 0.357270482, 0.0974844123, -0.406512678, -0.410135348, -0.0361238633, 0.319557907], [0.0330968207, -0.242239782, 0.120050196, 0.354699208, -0.259449241, -0.111198796, 0.408178231, -0.49369169, -0.0945378956, -0.200999891], [-0.00508218711, 0.0279895119, 0.10854607, -0.205785802, -0.327064799, -0.322969543, 0.20158026, -0.183918531, -0.0173896693, -0.0577639622], [-0.330439476, -0.347850159, 0.131036776, 0.361445303, -0.221495378, 0.0464919904, -0.124901835, -0.118702037, -0.159553208, -0.162154745], [-0.201710832, 0.352096879, -0.0853524652, -0.303965123, 0.248652839, 0.204120135, -0.355815628, -0.307002001, 0.130403792, 0.0500255052], [-0.35565632, -0.358085847, 0.375174217, 0.0531498801, -0.260087483, -0.275112154, 0.36165234, -0.216383649, 0.0667093926, -0.201697084], [0.356232481, -0.0961388357, -0.549493403, 0.0778573627, 0.24105018, 0.00970900917, -0.408791751, -0.488938923, 0.385680725, 0.0889963792], [0.143362353, 0.269327718, 0.244787555, -0.35992011, 0.135852254, 0.281867503, -0.396941322, 0.0197254158, -0.258052688, 0.0602810106], [0.627008761, -0.391383021, 0.0987911522, -0.214204531, -0.462414512, 0.348153174, 0.26478322, -0.289615151, 0.411229312, -0.256183171], [-0.154264597, -0.285642232, -0.155072304, -0.0137128291, 0.227392554, -0.03160814, 0.283170466, -0.143009119, 0.300754136, 0.404703713], [0.51172553, 0.345674948, -0.168641678, -0.447516003, 0.390521218, 0.277041254, -0.138741859, -0.277530649, -0.303700984, -0.41398118], [0.323175382, -0.273431957, -0.5496031, 0.0171143584, -0.345421225, 0.122412215, 0.0376235164, -0.127483756, -0.0572489809, 0.123600109], [0.0834735925, 0.235917647, 0.389529098, 0.207584558, 0.206365855, -0.207929, -0.0733902397, 0.253262386, -0.0108261534, 0.0402090516], [0.0918538087, 0.11121588, -0.0448296571, -0.290136507, 0.181043932, 0.111005524, 0.395251062, -0.0396308589, 0.18772069, -0.273251526], [-0.123382752, 0.0785606104, 0.34950883, 0.138226709, 0.0622882369, 0.218271075, 0.182346738, 0.00161043982, -0.110707991, 0.276548724], [-0.359003568, 0.524227274, -0.464293342, 0.48656788, 0.357516675, 0.442634409, -0.252647243, -0.0336858972, 0.100067501, -0.412973048], [-0.456645698, 0.439184205, 0.119949546, 0.242876404, 0.200820416, -0.430577944, -0.157195488, 0.225822032, -0.0569003748, -0.0749437329], [0.3114808, 0.346878706, 0.366319534, -0.434311019, 0.338656536, 0.544177463, 0.0742169998, 0.0397025143, -0.0941049381, -0.191869937], [-0.0594379928, -0.340265468, -0.414431659, 0.0366362306, -0.412025423, 0.245281056, 0.530689258, 0.18697425, 0.267614283, 0.0890690411], [0.230931018, -0.0261364425, -0.456325919, 0.315799262, 0.256000925, -0.309998167, -0.262697847, 0.289961611, 0.232436234, 0.296937078], [0.204664895, 0.209268703, 0.00848254576, -0.118940413, 0.176094717, -0.289193934, 0.221787663, -0.133881353, -0.0242023005, 0.0574290744], [-0.0473245363, -0.271759281, 0.0724203054, -0.00259983286, -0.321219763, -0.457908972, 0.391458294, -0.23689362, -0.0876132736, 0.355017692], [0.348973896, -0.477036431, -0.338384382, -0.191996773, -0.177403716, -0.0366580532, 0.218086165, 0.00215052599, 0.213229808, -0.236188527], [0.420362062, -0.100943114, 0.312014387, 0.37655519, 0.146052563, 0.00915329559, -0.318045374, 0.189355642, -0.282821249, -0.190349678], [-0.464377163, -0.0734003929, -0.261534955, -0.101290224, 0.361594678, -0.367261595, 0.0718936295, 0.0676878537, 0.151060124, -0.153885729], [0.103254185, 0.295547845, 0.326098668, -0.0785777503, 0.269772265, -0.385591062, -0.199843328, 0.168617518, 0.0165909257, 0.279216537], [-0.128924664, 0.243935407, 0.278280117, 0.581496363, 0.173386516, 0.0134701831, -0.325170483, 0.178089669, -0.115910912, 0.312991128], [-0.331263364, -0.422664329, 0.339366959, -0.169648744, -0.0351177282, 0.431587811, -0.520049928, 0.254091396, 0.227592041, -0.243991877], [0.35052986, -0.571905205, -0.471107833, -0.533590919, -0.449126403, 0.143094746, 0.140232166, 0.456657519, -0.140274045, 0.0617656333], [0.329512636, -0.243427067, -0.301335737, 0.31938069, 0.27593956, 0.0197571883, 0.348511645, 0.251733971, 0.287900855, 0.0945944405], [0.176657992, 0.340174993, 0.305554506, -0.235374512, -0.230240638, -0.0259766836, 0.148131053, -0.20486968, -0.245045753, -0.109292615], [-0.400005446, 0.153245577, -0.312541248, -0.137167233, 0.229961715, -0.199562677, -0.27627822, -0.0879451448, -0.162585002, 0.325665041], [0.364015779, 0.457426754, 0.160140936, -0.36871877, -0.196055368, 0.11944018, -0.316332548, -0.0728642211, 0.314638763, -0.0848101755], [-0.243985442, -0.0177158341, -0.0198541523, 0.2086722, -0.124525711, -0.0230356922, -0.495774754, -0.401406464, 0.425452164, 0.349288814], [-0.218235099, 0.296708985, 0.0980161799, -0.486208558, -0.354832477, -0.195496307, -0.489410277, 0.490566423, -0.241899988, -0.184538677], [-0.113389964, 0.331804359, 0.00578726207, 0.362895323, 0.288980581, 0.460011336, 0.0914215805, -0.0231043329, -0.323806926, -0.445252331], [0.374268442, -0.106529818, 0.182022992, -0.548487995, -0.216337494, -0.365513201, 0.24707248, 0.0139049757, -0.286353831, 0.105873063], [-0.192893113, -0.0737124657, 0.127860374, 0.530300424, -0.0618453289, 0.0802056374, 0.361243146, -0.399646831, -0.333910206, 0.0643338333]])
targets  = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0]).reshape(-1)
matching = np.array([[0.9178436, 0.25769881, 0.7951604], [0.78043125, 0.5019076, 0.7911015], [0.18629525, 0.98264458, 0.25982452], [0.29042939, 0.68576042, 0.76633173], [0.78591483, 0.01236143, 0.90869882], [0.97171503, 0.10690087, 0.25087941], [0.79621753, 0.84162531, 0.09957807], [0.3276865, 0.13511101, 0.59109914], [0.27643749, 0.2112022, 0.79536725], [0.52933932, 0.40406415, 0.42880976], [0.65270351, 0.30134413, 0.09982006], [0.28860319, 0.16267104, 0.6752936], [0.38387197, 0.85700311, 0.17560265], [0.02540087, 0.3045128, 0.96458696], [0.87064587, 0.6579779, 0.77603972], [0.95026791, 0.26443266, 0.84152384], [0.08308568, 0.59340681, 0.25779711], [0.92013597, 0.62944635, 0.51854748], [0.06603446, 0.50962861, 0.63852833], [0.94440443, 0.2006846, 0.61496476], [0.35649838, 0.07270849, 0.41351968], [0.9319572, 0.47673842, 0.36524826], [0.23304982, 0.08982598, 0.13011686], [0.60284726, 0.63857672, 0.03019988], [0.95089715, 0.43365123, 0.639663], [0.80763703, 0.02478772, 0.64585011], [0.52767556, 0.28419809, 0.17940561], [0.38363177, 0.78227761, 0.00116156], [0.41099913, 0.91637228, 0.70369362], [0.98361259, 0.50084461, 0.82470472], [0.79336703, 0.46931619, 0.85623882], [0.84766735, 0.21796204, 0.01196329], [0.91258467, 0.7556584, 0.03342948], [0.72508365, 0.40346992, 0.70542081], [0.51978679, 0.60088304, 0.24514274], [0.92122648, 0.87589922, 0.30879543], [0.92357521, 0.91941655, 0.5881349], [0.93358434, 0.17435692, 0.59300164], [0.7079047, 0.36417353, 0.1525478], [0.90895671, 0.8245928, 0.09320228], [0.45326851, 0.31794643, 0.3980996], [0.82312284, 0.9242727, 0.87996131], [0.18480065, 0.61526614, 0.67433083], [0.00551378, 0.41693243, 0.01008362], [0.86318825, 0.37032191, 0.43965085], [0.81672074, 0.48084287, 0.31795883], [0.77221628, 0.66085223, 0.18007598], [0.29219377, 0.56171722, 0.22133257], [0.47779518, 0.07948649, 0.42730857]])
np.save(features_path, features)
np.save(targets_path, targets)
np.save(matching_path, matching)

# Sample data
global_n_train = 6
global_n_val = 2
global_n_test = 2
global_do_stratify = False
global_seed = 0

def test_generate_random_split():
    # Sample data
    y = np.arange(1000)
    n_train = 700
    n_val = 150
    n_test = 150
    seed = 0
    do_stratify = False

    split = generate_random_split(y, n_train, n_val, n_test, do_stratify, seed, mask = False)

    assert len(split["idx_train"]) == n_train
    assert len(split["idx_val"]) == n_val
    assert len(split["idx_test"]) == n_test

    assert split["samplesize"] == n_train
    assert split["seed"] == seed
    assert split["stratify"] == do_stratify

    # Check the indices are within the range
    assert np.all((split["idx_train"] >= 0) & (split["idx_train"] < len(y)))
    assert np.all((split["idx_val"] >= 0) & (split["idx_val"] < len(y)))
    assert np.all((split["idx_test"] >= 0) & (split["idx_test"] < len(y)))


def test_generate_matched_split():
    x = np.load(features_path)
    y = np.load(targets_path).reshape(-1)
    matching = np.load(matching_path)

    x_mask = np.all(np.isfinite(x), 1)
    y_mask = np.isfinite(y)
    xy_mask = np.logical_and(x_mask, y_mask)

    # if there are more than 1 matching variable (e.g. age, gender, ed level)
    if len(matching) == len(y) and len(matching.shape) > 1:
        m_mask = np.all(np.isfinite(matching), 1)
        xy_mask = np.logical_and(xy_mask, m_mask)
    
    # preparing for matching when there's only one matching variable
    elif len(matching) == len(y) and len(matching.shape) == 1:
        m_mask = np.isfinite(matching)
        xy_mask = np.logical_and(xy_mask, m_mask)
    
    split_dict = generate_matched_split(
        y = y,
        match = matching,
        n_train = global_n_train,
        n_val = global_n_val,
        n_test = global_n_test,
        do_stratify = global_do_stratify,
        mask = xy_mask,
        seed = global_seed)

    expected_dict =  {'idx_train': np.array([42, 44, 22,  3, 24, 20]), 'idx_val': np.array([37, 19]), 'idx_test': np.array([31, 10]), 'samplesize': 6, 'seed': 0, 'stratify': False, "average_matching_score": 0.534138759114014}

    assert (split_dict["idx_train"] == expected_dict["idx_train"]).all()
    assert (split_dict["idx_val"] == expected_dict["idx_val"]).all()
    assert (split_dict["idx_test"] == expected_dict["idx_test"]).all()
    assert split_dict["samplesize"] == expected_dict["samplesize"]
    assert split_dict["seed"] == expected_dict["seed"]
    assert split_dict["stratify"] == expected_dict["stratify"]
    assert split_dict["average_matching_score"] == expected_dict["average_matching_score"]

def test_write_splitfile():
    # ###
    # # Test case 1: sampling_type == "none", generate_random_split is called
    # ###
    sampling_type = "none"

    write_splitfile(
        features_path,
        targets_path,
        split_path,
        matching_path,
        sampling_type,
        global_n_train,
        global_n_val,
        global_n_test,
        global_seed,
        global_do_stratify
    )

    with open(split_path, "r") as f:
        split_dict = json.load(f)

    expected_dict = {'idx_train': np.array([38, 34, 22, 27, 45, 1]), 'idx_val': np.array([35, 43]), 'idx_test': np.array([29, 4]), 'samplesize': 6, 'seed': 0, 'stratify': False}

    # verify generate_random_split function is called
    assert (split_dict["idx_train"] == expected_dict["idx_train"]).all()
    assert (split_dict["idx_val"] == expected_dict["idx_val"]).all()
    assert (split_dict["idx_test"] == expected_dict["idx_test"]).all()
    assert split_dict["samplesize"] == expected_dict["samplesize"]
    assert split_dict["seed"] == expected_dict["seed"]
    assert split_dict["stratify"] == expected_dict["stratify"]
    assert "average_matching_score" not in split_dict

    # Remove temporary files
    for file in [split_path]:
        if(os.path.exists(file) and os.path.isfile(file)):
            os.remove(file)


    ###
    # Test case 2: sampling_type == file path, generate_matched_split is called
    ###
    sampling_type = matching_path

    write_splitfile(
        features_path,
        targets_path,
        split_path,
        matching_path,
        sampling_type,
        global_n_train,
        global_n_val,
        global_n_test,
        global_seed,
        global_do_stratify
    )

    with open(split_path, "r") as f:
        split_dict = json.load(f)

    expected_dict = {'idx_train': np.array([42, 44, 22, 3, 24, 20]), 'idx_val': np.array([37, 19]), 'idx_test': np.array([31, 10]), 'samplesize': 6, 'seed': 0, 'stratify': False, 'average_matching_score': 0.534138759114014}

    # verify generate_random_split function is called
    assert (split_dict["idx_train"] == expected_dict["idx_train"]).all()
    assert (split_dict["idx_val"] == expected_dict["idx_val"]).all()
    assert (split_dict["idx_test"] == expected_dict["idx_test"]).all()
    assert split_dict["samplesize"] == expected_dict["samplesize"]
    assert split_dict["seed"] == expected_dict["seed"]
    assert split_dict["stratify"] == expected_dict["stratify"]
    assert split_dict["average_matching_score"] == expected_dict["average_matching_score"]

    # Remove temporary files
    for file in [features_path, targets_path, matching_path, split_path]:
        if(os.path.exists(file) and os.path.isfile(file)):
            os.remove(file)
