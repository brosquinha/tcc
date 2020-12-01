from dataset.collectors.multi_tweets_collector import MultiTweetsCollector
from dataset.collectors.tweet_collector import TweetCollector
from dataset.twitter_data import TwitterData


class RepliesCollector(TweetCollector):
    """
    Handles obtaining replies to tweet
    """

    def __init__(self):
        super().__init__()
        self.twitter = TwitterData(selenium=True, headless=True, login=True)
        self.multi_tweets_collector = MultiTweetsCollector()
        self.multi_tweets_collector.reply_id = True
    
    def get_tweet(self, tweet_id):
        self.tweet = self._db_has_tweet_id(self.tweet_db, tweet_id)
        if self.tweet:
            super().get_tweet(tweet_id)
        else:
            self.logger.warning("Tweet %s not found" % tweet_id)
    
    def _request_tweet(self, tweet_id):
        tweets_ids = self.twitter.get_replies_to_tweet(self.tweet.user_name, tweet_id)
        self.multi_tweets_collector.reply_id = tweet_id
        for tid in tweets_ids:
            self.multi_tweets_collector.get_tweet(tid)
        self.multi_tweets_collector.wrap_up()
