import csv
import os

import matplotlib.pyplot as plt
import numpy as np

from emotion_classification.sem_eval_naive_bayes import SingleEmotionSemEvalNaiveBayes
from emotion_classification.tec_naive_bayes import SingleEmotionTECNaiveBayes
from emotion_classification.text_emotion_naive_bayes import SingleEmotionTextEmotionNaiveBayes
from dataset.model.tweet_model import TweetModel
from dataset.tweet_db import TweetDB

emotions = ['anger', 'fear', 'joy', 'sadness']

def predict_replies(filepath, **kwargs):
    identified_replies = []
    files_tec = [os.path.join('output', '%s-tec.pickle' % e) for e in emotions]
    for i, f in enumerate(files_tec):
        # nb = SingleEmotionSemEvalNaiveBayes(emotion_index=i, filename=f)
        nb = SingleEmotionTECNaiveBayes(emotion_name=emotions[i], filename=f)
        tdb = TweetDB()
        tweets = list(tdb.all_replies())
        tweets_with_emotion_indexes, total = nb.classify_sentences([t.text for t in tweets])
        identified_replies.append(tweets_with_emotion_indexes)
        print('%d of %d on %s category' % (len(tweets_with_emotion_indexes), total, emotions[i]))

    # Count how many additional emotions each tweet has, and prints to the console the average and median for each emotion
    for index, tweets_indexes in enumerate(identified_replies):
        other_tweets_indexes = [ti for tis in identified_replies for ti in tis if tis != tweets_indexes]
        repeated_tweets_indexes = [other_tweets_indexes.count(ti) for ti in tweets_indexes if ti in other_tweets_indexes]
        one_emotion_tweets = [tweets[ti] for ti in tweets_indexes if ti not in other_tweets_indexes]
        print('%s: %d out of %d have been identified with another emotion as well (avg: %.2f, mdn: %.2f)' % (
            emotions[index], len(repeated_tweets_indexes), len(tweets_indexes),
            np.mean(repeated_tweets_indexes), np.median(repeated_tweets_indexes)
        ))

        print('%d tweets have been identified only with %s' % (
            (len(tweets_indexes) - len(repeated_tweets_indexes)), emotions[index]))

        get_parent_tweets_metrics(one_emotion_tweets, emotions[index])

    count_repeated_emotions(tweets, identified_replies)

def count_repeated_emotions(tweets, identified_replies):
    result = [[], [], [], [], []]
    
    flattened_idenfitied_replies = [ti for tis in identified_replies for ti in tis]
    for index in range(len(tweets)):
        if index not in flattened_idenfitied_replies:
            result[0].append(tweets[index])
        elif flattened_idenfitied_replies.count(index) == 1:
            result[1].append(tweets[index])
        elif flattened_idenfitied_replies.count(index) == 2:
            result[2].append(tweets[index])
        elif flattened_idenfitied_replies.count(index) == 3:
            result[3].append(tweets[index])
        elif flattened_idenfitied_replies.count(index) == 4:
            result[4].append(tweets[index])
            
    print('Tweets that have not been identified with any emotion whatsoever: %d' % len(result[0]))
    
    # Display how many tweets have only one emotion
    print('Tweets that have only 1 emotion identified: %d' % len(result[1]))
    
    # Display how many tweets have n emotions
    for n in range(2, len(result)):
        print('Tweets that have %d emotions identified: %d' % (n, len(result[n])))
    
    return result

def get_parent_tweets_metrics(tweets_with_emotion, emotion):
    tdb = TweetDB()
    parent_tweet = lambda tid: tdb.get_by(str(tid))
    parents_tweets = list(set([parent_tweet(tweet.parent_tweet_id) for tweet in tweets_with_emotion]))
    parents_retweets_count = [int(t.retweet_count) for t in parents_tweets]
    parents_retweets_count_median = np.median(np.array(parents_retweets_count)) if parents_retweets_count else 0
    parents_retweets_75_top = np.percentile(np.array(parents_retweets_count), 75)
    parents_retweets_90_top = np.percentile(np.array(parents_retweets_count), 90)
    parents_retweets_95_top = np.percentile(np.array(parents_retweets_count), 95)
    parents_retweets_count_sorted = np.sort(parents_retweets_count)[::-1]
    parents_retweets_count_top_20_percentage_len = int(len(parents_retweets_count) * 0.2)
    parents_retweets_count_top_20_percentage = parents_retweets_count_sorted[:parents_retweets_count_top_20_percentage_len]
    print('%d on %s category' % (len(tweets_with_emotion), emotion))
    print('%d spreader tweets with retweets median: %.2f, 75%% percentile: %.2f, 90%% percentile: %.2f, 95%% percentile: %.2f, and responses average: %.2f)' % (
        len(parents_tweets),
        parents_retweets_count_median, parents_retweets_75_top, parents_retweets_90_top, parents_retweets_95_top,
        len(tweets_with_emotion) / len(parents_tweets)
        )
    )
    print('top 20%%: avg: %.2f, median: %.2f' % (
        np.mean(parents_retweets_count_top_20_percentage),
        np.median(parents_retweets_count_top_20_percentage)
    ))

def predict_top_retweeted_fake_news_tweets(models: list, consolidate=2):
    number_of_tweets_to_evaluate = 500
    tdb = TweetDB()
    # most_popular_tweets = list(tdb.all_sorted_by(sort=TweetModel.retweet_count.desc()))
    most_popular_tweets = list(tdb.all_sorted_by_eager_loading(sort=TweetModel.retweet_count.desc()))
    classified_init = False

    get_model_name = lambda model: model.__name__.replace("SingleEmotion", "")

    with open('output/popular_tweets.csv', 'w', newline='') as f:
        csv_writer = csv.writer(f)
        
        csv_top_row = [
            'TweetID', 'Tweet retweet count', 'Tweet replies count', 'Tweet source',
        ]
        for model, _ in models:
            for emotion in emotions:
                csv_top_row.append(f'{get_model_name(model)} {emotion}')
            for emotion in emotions:
                csv_top_row.append(f'{get_model_name(model)} {emotion}%')
        if consolidate:
            for emotion in emotions:
                csv_top_row.append(f'Consolidation {emotion}')
            for emotion in emotions:
                csv_top_row.append(f'Consolidation {emotion}%')
        csv_writer.writerow(csv_top_row)

        model_instances = {}
        for model, filename_template in models:
            model_instances[get_model_name(model)] = []
            for emotion in emotions:
                model_instances[get_model_name(model)].append(
                    model(emotion_name=emotion, filename=filename_template.format(emotion))
                )

        tweet_count = 0
        for tweet in most_popular_tweets:
            if tweet_count > number_of_tweets_to_evaluate:
                break
            replies = tweet.replies
            if not len(replies):
                continue
            
            tweet_count += 1
            print(f'Found {len(replies)} for tweet {tweet.id} with {tweet.retweet_count} retweets')
            
            datasets_lists = {}
            csv_row = [tweet.id, tweet.retweet_count, len(replies), tweet.source]
            for model_name in model_instances.keys():
                num_replies_with_emotions = []

                for model_instance in model_instances[model_name]:
                    tweets_with_emotion_indexes, _ = model_instance.classify_sentences([t.text for t in replies])
                    num_replies_with_emotions.append(tweets_with_emotion_indexes)

                csv_row += [len(x) for x in num_replies_with_emotions]
                csv_row += [len(x) / len(replies) for x in num_replies_with_emotions]
                datasets_lists[model_name] = num_replies_with_emotions

            if consolidate:
                consolidate_results = []
                for index_emotion, _ in enumerate(emotions):
                    consolidate_results.append(
                        consolidate_classifiers(
                            consolidate, replies, [x[index_emotion] for x in datasets_lists.values()]
                        )
                    )
                csv_row += [len(x) for x in consolidate_results]
                csv_row += [len(x) / len(replies) for x in consolidate_results]
                datasets_lists['Consolidate'] = consolidate_results
            
            csv_writer.writerow(csv_row)

            save_classified_replies(replies, emotions, datasets_lists, classified_init)
            classified_init = True

def save_classified_replies(replies, emotions, datasets, init):
    with open('output/popular_classified_replies.csv', 'a' if init else 'w',  encoding='utf-8', newline="") as f:
        csv_writer = csv.writer(f)
        if not init:
            csv_header_row = ['TweetID', 'Tweet text', 'Parent tweet id']
            for dataset_name in datasets.keys():
                for emotion in emotions:
                    csv_header_row.append(f'{dataset_name} {emotion}')
            csv_writer.writerow(csv_header_row)
    
        for index_reply, reply in enumerate(replies):
            csv_row = [reply.id, reply.text, reply.parent_tweet_id]
            for dataset_name, dataset_list in datasets.items():
                for index_emotion, emotion in enumerate(emotions):
                    csv_row.append(index_reply in dataset_list[index_emotion])
            csv_writer.writerow(csv_row)

def consolidate_classifiers(consolidate_number, tweets, classified_tweets_lists):
    classified_tweets_sets = [set(l) for l in classified_tweets_lists]
    consolidated_list = []
    for index, _ in enumerate(tweets):
        count = len([x for x in classified_tweets_sets if index in x])
        if count >= consolidate_number:
            consolidated_list.append(index)
    return consolidated_list

def plot_top_retweeted_tweets_analisys(dataset_and_model, poly_deg=2, min_range=300, max_range=10000, outlier_const=10.5):
    translate_dict = {
        'naive-bayes': {
            'semeval': 'SemEvalNaiveBayes',
            'tec': 'TECNaiveBayes',
            'text-emotion': 'TextEmotionNaiveBayes',
            'consolidation': 'Consolidation'
        },
        'lstm': {
            'semeval': 'SemEvalLSTM',
            'tec': 'TECLSTM',
            'text-emotion': 'TextEmotionLSTM',
            'consolidation': 'Consolidation'
        }
    }
    dataset, model = dataset_and_model
    
    with open('output/popular_tweets.csv') as f:
        data = csv.reader(f)
        data = list(data)
        top_row = data[0]
        data = data[1:]
    
    try:
        anger_index = top_row.index(f'{translate_dict[model][dataset]} anger%')
        fear_index = top_row.index(f'{translate_dict[model][dataset]} fear%')
        joy_index = top_row.index(f'{translate_dict[model][dataset]} joy%')
        sadness_index = top_row.index(f'{translate_dict[model][dataset]} sadness%')
    except ValueError:
        print("Invalid dataset")
        return
    
    # Filter data to have only elements with min_range <= rt_count <= max_range 
    data = [x for x in data if int(x[1]) in range (min_range, max_range + 1)]

    x_data = [int(x[1]) for x in data]
    anger_data = [float(x[anger_index]) for x in data]
    fear_data = [float(x[fear_index]) for x in data]
    joy_data = [float(x[joy_index]) for x in data]
    sadness_data = [float(x[sadness_index]) for x in data]
 
    x_anger_data = x_data
    x_fear_data = x_data
    x_joy_data = x_data
    x_sadness_data = x_data
    
    x_anger_data, anger_data = _reject_outliers(x_anger_data, anger_data, outlier_const)
    x_fear_data, fear_data = _reject_outliers(x_fear_data, fear_data, outlier_const)
    x_joy_data, joy_data = _reject_outliers(x_joy_data, joy_data, outlier_const)
    x_sadness_data, sadness_data = _reject_outliers(x_sadness_data, sadness_data, outlier_const)

    anger_trendline = np.polyfit(x_anger_data, anger_data, poly_deg)
    anger_trendline = np.poly1d(anger_trendline)(x_anger_data)
    fear_trendline = np.polyfit(x_fear_data, fear_data, poly_deg)
    fear_trendline = np.poly1d(fear_trendline)(x_fear_data)
    joy_trendline = np.polyfit(x_joy_data, joy_data, poly_deg)
    joy_trendline = np.poly1d(joy_trendline)(x_joy_data)
    sadness_trendline = np.polyfit(x_sadness_data, sadness_data, poly_deg)
    sadness_trendline = np.poly1d(sadness_trendline)(x_sadness_data)

    plt.figure(figsize=[8.4, 4.8])
    ax = plt.gca()
    ax.grid(linestyle='--', alpha=0.5)
    ax.scatter(x_anger_data, anger_data, s=10, c='r', alpha=0.8, label='Raiva %')
    ax.scatter(x_fear_data, fear_data, s=10, c='g', alpha=0.8, label='Medo %')
    ax.scatter(x_joy_data, joy_data, s=10, c='y', alpha=0.8, label='Alegria %')
    ax.scatter(x_sadness_data, sadness_data, s=10, c='b', alpha=0.8, label='Tristeza %')
    ax.plot(x_anger_data, anger_trendline, 'r--', alpha=0.5)
    ax.plot(x_fear_data, fear_trendline, 'g--', alpha=0.5)
    ax.plot(x_joy_data, joy_trendline, 'y--', alpha=0.5)
    ax.plot(x_sadness_data, sadness_trendline, 'b--', alpha=0.5)
    ax.set_xscale('log')
    ax.set_xlim(min_range, max_range)
    ax.set_ylim(0, 1)
    plt.xlabel('Quantidade de retweets')
    plt.ylabel('Proporção de respostas com emoção')
    plt.title(f'Popularidade x emoções provocadas ({dataset} {model})')
    ax.legend(loc='upper right')
    plt.savefig(f'output/popular_tweets_{dataset.replace("-", "_")}_{model}.png')
    plt.close()

def _reject_outliers(x_data, y_data, m):
    mean = np.mean(y_data)
    std_deviation = np.std(y_data)
    print(std_deviation)
    deleted_members = 0
    x_elements_to_delete = []
    y_elements_to_delete = []
    for index, y in enumerate(y_data):
        if abs(y - mean) > m * std_deviation:
            x_elements_to_delete.append(index)
            y_elements_to_delete.append(index)
    x_data = np.delete(x_data, x_elements_to_delete, 0)
    y_data = np.delete(y_data, y_elements_to_delete, 0)
    
    return (x_data, y_data)
