from dataset.collectors.tweet_collector import TweetCollector


class RetweetsCollector(TweetCollector):
    """
    Handles obtaining retweets to tweet
    """

    def _request_tweet(self, tweet_id):
        return self.twitter.get_retweets_to_tweet(tweet_id)
