import csv
import sys
from typing import List

from dataset.collectors.multi_tweets_collector import MultiTweetsCollector
from dataset.collectors.replies_collector import RepliesCollector
from dataset.collectors.retweets_collector import RetweetsCollector
from utils.loggable import Loggable


class FakeNewsNetCollector(Loggable):
    """
    Class for collecting tweets from FakeNewsNet's CSV files
    """
    
    _valid_types = {
        'tweet': MultiTweetsCollector,
        'retweet': RetweetsCollector,
        'reply': RepliesCollector
    }
    
    def __init__(self, filelist: list):
        csv.field_size_limit(sys.maxsize)
        self.filelist = filelist
        super().__init__()

    def collect(self, collect_types: List[str]) -> None:
        """
        Collects all tweets of given types

        Valid tweet types are:
            * tweet: get all tweets in CSV files
            * retweet: get all retweets to tweets in CSV files
            * reply: get all replies to tweets in CSV files

        :param collect_types: Types to collect tweets
        :type collect_types: List[str]
        """
        valid_types = [x for x in collect_types if x in self._valid_types.keys()]
        for ctype in valid_types:
            self._collect_tweets(ctype)

    @staticmethod
    def get_valid_types():
        return FakeNewsNetCollector._valid_types.keys()
    
    def _collect_tweets(self, tweet_type: str):
        collector: TweetCollector = self._valid_types[tweet_type]()
        self._get_last_seen_tweet(tweet_type)
        try:
            for tweets_page in self.get_tweets_ids_from_csv():
                for tweet_id in tweets_page:
                    if not tweet_id or self._is_already_scanned_tweet(tweet_id):
                        continue
                    self.last_tweet_id = None
                    self.logger.info("Requiring %s" % tweet_id)
                    collector.get_tweet(tweet_id)
            collector.wrap_up()
        except KeyboardInterrupt:
            self._save_last_seen_tweet(tweet_type, collector.get_last_tweet_id() or tweet_id)
            raise
        finally:
            collector.close()
    
    def get_tweets_ids_from_csv(self):
        """
        Gets all Tweet IDs from source CSV files
        """
        for fname in self.filelist:
            with open(fname) as f:
                fakenews = csv.reader(f)
                next(fakenews) # Discard top CSV row
                for fake in fakenews:
                    yield fake[3].split("	")

    def _is_already_scanned_tweet(self, current_id):
        return self.last_tweet_id and str(self.last_tweet_id) != str(current_id)

    def _get_last_seen_tweet(self, tweet_type, ):
        try:
            with open('last_tweet_id_%s.txt' % tweet_type) as f:
                self.last_tweet_id = f.read().strip()
            self.logger.info("Continuing from %s" % self.last_tweet_id)
        except:
            self.last_tweet_id = None

    def _save_last_seen_tweet(self, tweet_type, last_seen_tweet):
        with open('last_tweet_id_%s.txt' % tweet_type, 'w') as f:
            f.write(last_seen_tweet)
        self.logger.info("Will resume from %s" % last_seen_tweet)
