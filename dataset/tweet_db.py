import logging

from sqlalchemy.orm import joinedload

from dataset.model.tweet_model import TweetModel
from dataset.base import Session
from utils.loggable import Loggable


class TweetDB(Loggable):
    
    def __init__(self):
        self.session = Session()
        super().__init__()

    def save_tweet(self, tweet: dict):
        tweet_model = TweetModel()
        tweet_model.load_from_dict(tweet)
        self.session.add(tweet_model)

    def get_by(self, tweet_id: str):
        return self.session.query(TweetModel).get(tweet_id)

    def all_by(self, **kwargs):
        return self.session.query(TweetModel).filter_by(**kwargs).yield_per(100)

    def all_sorted_by(self, sort, **kwargs):
        return self.session.query(TweetModel).filter_by(**kwargs).order_by(sort).yield_per(100)

    def all_sorted_by_eager_loading(self, sort, **kwargs):
        return self.session.query(TweetModel).filter_by(**kwargs).options(joinedload('replies')).order_by(sort)

    def all_replies(self, **kwargs):
        return self.session.query(TweetModel).filter(TweetModel.parent_tweet_id.isnot(None)).filter_by(**kwargs).yield_per(100)

    def all_replies_like(self, search_text, **kwargs):
        return self.session.query(TweetModel).filter(
            TweetModel.parent_tweet_id.isnot(None), TweetModel.text.like(f'%{search_text}%')).filter_by(**kwargs).yield_per(100)

    def commit(self):
        self.session.commit()

    def close(self):
        self.session.close()
        self.logger.debug("Session closed")
    
    def __enter__(self):
        return self

    def __exit__(self, *args, **kwargs):
        self.close()
