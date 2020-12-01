from sqlalchemy import BigInteger, Column, ForeignKey, Integer, String
from sqlalchemy.orm import relationship, backref

from dataset.base import Base


class TweetModel(Base):
    __tablename__ = 'tweet_model'
    id = Column(BigInteger, primary_key=True, nullable=False)
    text = Column(String, nullable=False)
    retweet_count = Column(Integer, nullable=False)
    favorite_count = Column(Integer, nullable=False)
    user_name = Column(String, nullable=False)
    user_id = Column(BigInteger, nullable=False)
    source = Column(String)
    parent_tweet_id = Column(BigInteger, ForeignKey('tweet_model.id'), nullable=True)
    replies = relationship("TweetModel", remote_side=[parent_tweet_id])

    def load_from_dict(self, tweet):
        self.id = tweet['id']
        self.text = tweet['full_text']
        self.retweet_count = tweet['retweet_count']
        self.favorite_count = tweet['favorite_count']
        self.user_name = tweet['user']['screen_name']
        self.user_id = tweet['user']['id']
        self.parent_tweet_id = tweet.get('in_reply_to_status_id')
