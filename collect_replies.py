from dataset.collectors.fake_news_net_collector import FakeNewsNetCollector
from dataset.model.tweet_model import TweetModel
from dataset.tweet_db import TweetDB


class RepliesOnlyCollector(FakeNewsNetCollector):

    def get_tweets_ids_from_csv(self):
        yield [str(t.id) for t in TweetDB().all_sorted_by(
            sort=TweetModel.retweet_count.desc(), source="politifact")]
        yield [str(t.id) for t in TweetDB().all_sorted_by(
            sort=TweetModel.retweet_count.desc(), source="gossipcop")]

if __name__ == "__main__":
    try:
        collector = RepliesOnlyCollector([])
        collector.collect(collect_types=["reply"])
    except KeyboardInterrupt:
        pass