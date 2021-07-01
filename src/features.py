from utils import load_trainset, load_testset
from model import FeatureMatcher


def train():
    images, indices, logo_labels = load_trainset()

    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    feature_matcher = FeatureMatcher(index_params, search_params)
    feature_matcher.train(images, indices, logo_labels)

    return feature_matcher


def test_and_evaluate(feature_matcher):
    images, indices, logo_labels = load_testset()
    feature_matcher.detect(images)


if __name__=='__main__':
    model = train()
    test_and_evaluate(model)


