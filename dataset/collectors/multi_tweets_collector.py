from dataset.collectors.tweet_collector import TweetCollector


class MultiTweetsCollector(TweetCollector):
    """
    Handles obtaining multi tweets at once
    """

    twitter_lookup_max_len = 100
    
    def __init__(self):
        self.tweets_chunck = []
        super().__init__()
    
    def get_tweet(self, tweet_id: str):
        if self._db_has_tweet_id(self.tweet_db, tweet_id):
            self.logger.info("Tweet %s already saved, skiping..." % tweet_id)
            return
        if len(self.tweets_chunck) < self.twitter_lookup_max_len - 1:
            self.tweets_chunck.append(tweet_id)
            self.logger.debug(tweet_id)
            return
        tweets = self._get_tweet(self.tweets_chunck + [tweet_id])['id'] or {}
        self._handle_tweets(tweets)

    def get_last_tweet_id(self):
        return self.tweets_chunck[0] if len(self.tweets_chunck) else False

    def wrap_up(self):
        if not self.tweets_chunck:
            return
        tweets = self._get_tweet(self.tweets_chunck)['id'] or {}
        self._handle_tweets(tweets)
        super().wrap_up()
    
    def _request_tweet(self, tweet_id):
        return self.twitter.get_tweets(tweet_id)

    def _handle_tweets(self, tweets):
        for tid, tweet in tweets.items():
            self._enqueue_tweet(tweet, tid)
        self.tweet_db.commit()
        self.tweets_chunck = []
