import os
import time

from dotenv import load_dotenv
from selenium import webdriver
from selenium.common.exceptions import InvalidSessionIdException, TimeoutException
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from twython import Twython

from utils.loggable import Loggable


class TwitterData(Loggable):

    def __init__(self, selenium=False, headless=True, login=True):
        load_dotenv()
        auth = Twython(
            app_key=os.getenv("TWITTER_API_KEY"),
            app_secret=os.getenv("TWITTER_API_SECRET"),
            oauth_version=2
        )
        self.api = Twython(
            app_key=os.getenv("TWITTER_API_KEY"), access_token=auth.obtain_access_token())
        
        self.driver = None
        self.selenium_login = login
        self.driver_headless = headless
        if selenium:
            self._init_selenium_driver()
            self._login_twitter()

        super().__init__()

    def get_tweet(self, tweet_id):
        return self.api.show_status(id=tweet_id, tweet_mode="extended")

    def get_tweets(self, tweet_id_list):
        return self.api.lookup_status(
            id=tweet_id_list, map=True, include_entities=True, tweet_mode="extended")

    def get_retweets_to_tweet(self, tweet_id):
        return self.api.get_retweets(id=tweet_id, tweet_mode="extended")

    def get_replies_to_tweet(self, username, tweet_id):
        if not self.driver:
            self._init_selenium_driver()
            self._login_twitter()
        
        try:
            again = 1
            while again > 0:
                assert again <= 5
                self.driver.get("https://twitter.com/{}/status/{}".format(username, tweet_id))
                again = 0
        except InvalidSessionIdException:
            self.logger.info("Invalid session ID, restarting Driver...")
            self.driver.quit()
            time.sleep(5)
            self._init_selenium_driver()
            self._login_twitter()
            again += 1
        except TimeoutException:
            self.logger.info("Timeout, sleeping for 30 seconds before retrying")
            time.sleep(30)
            again += 1
        except AssertionError:
            self.logger.error("Max attempts reached! Operation canceled")
        replies = set()

        try:
            while True:
                last_height = self.driver.execute_script("return document.body.scrollHeight")
                tweet_containers = self.driver.find_elements_by_css_selector("div[data-testid=\"tweet\"]")
                for container in tweet_containers:
                    links = container.find_elements_by_css_selector("a")
                    if len(links) > 2:
                        tid = links[2].get_attribute("href").split("/")[-1]
                        try:
                            int(tid)
                            replies.add(tid)
                        except ValueError:
                            self.logger.warn("%s is not a Tweet ID" % tid)
                self.driver.execute_script("window.scrollTo(0, window.scrollY + 200);")
                WebDriverWait(self.driver, 30).until(
                    lambda d: len(d.find_elements_by_css_selector('div[role="progressbar"]')) == 0
                )
                if self.driver.execute_script("return ((window.innerHeight + window.scrollY) >= document.body.scrollHeight)"):
                    break
        except TimeoutException:
            self.logger.error("Progressbar timeout, returning all %d captured tweets..." % len(replies))
        return replies

    def _init_selenium_driver(self):
        options = Options()
        if self.driver_headless:
            options.add_argument('--headless')
            options.add_argument('--disable-gpu')
            options.add_argument('window-size=1024,768')
        chromedriver_path = os.path.join(os.getcwd(), 'chromedriver')
        self.driver = webdriver.Chrome(executable_path=chromedriver_path, options=options)
        self.driver.implicitly_wait(3)

    def _login_twitter(self):
        if not self.selenium_login:
            return
        
        self.driver.get("https://twitter.com/login")

        self.driver.find_element_by_css_selector('input[name="session[username_or_email]"]').send_keys(os.getenv("TWITTER_USERNAME"))
        self.driver.find_element_by_css_selector('input[name="session[password]"]').send_keys(os.getenv("TWITTER_PASSWORD"))
        self.driver.find_element_by_css_selector('div[role="button"]').click()
        WebDriverWait(self.driver, 5).until(
            lambda d: 'Home' in d.title
        )


if __name__ == "__main__":
    twitter = TwitterData(headless=False, login=False)
    for retweet in twitter.get_replies_to_tweet("TedDBexar", "984247385235668992"):
        print(retweet)
