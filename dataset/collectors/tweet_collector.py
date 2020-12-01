import time
from abc import abstractmethod
from datetime import datetime

from twython.exceptions import TwythonRateLimitError

from dataset.model.tweet_model import TweetModel
from dataset.tweet_db import TweetDB
from dataset.twitter_data import TwitterData
from utils.loggable import Loggable


class TweetCollector(Loggable):
    """
    Helper class to handle API calls and database connections
    """
    
    def __init__(self):
        self.twitter = TwitterData()
        self.tweet_db = TweetDB()
        self.reply_id = False
        super().__init__()

    def get_tweet(self, tweet_id: str):
        tweets = self._get_tweet(tweet_id) or []
        self.logger.info("Got %d tweets for %s" % (len(tweets), tweet_id))
        for tweet in tweets:
            self._enqueue_tweet(tweet, tweet.get('id'))
        self.tweet_db.commit()

    def get_last_tweet_id(self):
        return False

    def wrap_up(self):
        self.logger.info("Finished collecting tweets")
    
    def close(self):
        try:
            self.tweet_db.commit()
        except:
            self.tweet_db.rollback()
        self.tweet_db.close()
        self.logger.debug("Queue completed")
    
    @abstractmethod
    def _request_tweet(self, tweet_id):
        pass
    
    def _db_has_tweet_id(self, db_conn, tweet_id) -> TweetModel:
        return db_conn.session.query(TweetModel).get(tweet_id)

    def _get_tweet(self, tweet_id):
        while True:
            try:
                return self._request_tweet(tweet_id)
            except TwythonRateLimitError as e:
                self._wait_retry_after(e.retry_after)
            except Exception as e:
                self.logger.exception(str(e))
                return None

    def _wait_retry_after(self, retry_after: str):
        self.logger.debug("Retry-after: %s" % retry_after)
        try:
            sleep_time = datetime.fromtimestamp(int(retry_after)) - datetime.now()
            sleep_time = sleep_time.total_seconds()
        except Exception as e:
            self.logger.debug(str(e))
            sleep_time = 60 * 15
        self.logger.info("Sleeping for %d seconds" % sleep_time)
        time.sleep(sleep_time)

    def _enqueue_tweet(self, tweet: dict, tid: str):
        if not tweet:
            self.logger.warning("Could not get tweet %s" % tid)
            return
        if self.reply_id:
            if str(tweet['in_reply_to_status_id']) != str(self.reply_id):
                self.logger.warning("Tweet %s not reply to %s, skiping..." % (tid, self.reply_id))
                return
        else:
            tweet['in_reply_to_status_id'] = None # Do not treat these tweets as replies, even if they are
        if self._db_has_tweet_id(self.tweet_db, tid):
            self.logger.info("Tweet %s already saved, skiping..." % tid)
        else:
            self.tweet_db.save_tweet(tweet)
            self.logger.info("Twitter %s saved" % tid)
