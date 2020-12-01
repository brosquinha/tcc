from argparse import ArgumentParser

from dataset.collectors.fake_news_net_collector import FakeNewsNetCollector

filelist = [
    './dataset/source/politifact_fake.csv', './dataset/source/gossipcop_fake.csv']

if __name__ == "__main__":
    parser = ArgumentParser(description='Collects tweets for emotion classification')
    parser.add_argument(
        'collect_types',
        help="What kind of tweets to collect",
        nargs='+',
        choices=FakeNewsNetCollector.get_valid_types()
    )
    args = parser.parse_args()
    try:
        collector = FakeNewsNetCollector(filelist)
        collector.collect(**vars(args))
    except KeyboardInterrupt:
        pass
