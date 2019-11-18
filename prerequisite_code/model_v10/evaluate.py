from __future__ import division
from sklearn import metrics



def evaluate(predictions, config):
    '''
    :param predictions: file_key = example["pre_topic"] + "_" + example["post_topic"]
    predictions[file_key] = [final_score, original_label, example["pre_topic"], example["post_topic"], pre_topic_pre_mention_to_predicted, pre_topic_post_mention_to_predicted,
                                     post_topic_pre_mention_to_predicted, post_topic_post_mention_to_predicted]
    :param config:
    :return: precision, recall, f
    '''

    '''
    True Positive（TP）：预测为正例，实际为正例

    False Positive（FP）：预测为正例，实际为负例

    True Negative（TN）：预测为负例，实际为负例

    False Negative（FN）：预测为负例，实际为正例
    '''

    pred = []   #预测结果
    true = []   #实际标签
    TP = 0
    FP = 0
    TN = 0
    FN = 0
    with open(config["eval_result_file"], 'w') as eval_file:
        for file_key, result in predictions.items():
            score = result[0]
            label = result[1]
            pre_topic = result[2]
            post_topic = result[3]
            #print("pre topic: " + pre_topic + '\t' + "post topic: " + post_topic + "\t" + "predicted score: " + str(score) + '\t' + "true label: " + str(label))
            eval_file.write("pre topic: " + pre_topic + '\t' + "post topic: " + post_topic + "\t" + "predicted score: " + str(score) + '\t' + "true label: " + str(label) + '\n'
                            + "pre_topic_pre_mention_to_predicted: " + str(result[4]) + '\n'+ "pre_topic_post_mention_to_predicted: " + str(result[5]) + '\n'
                            + "post_topic_pre_mention_to_predicted: " + str(result[6]) + '\n' + "post_topic_post_mention_to_predicted: " + str(result[7]) + "\n")
            if score > config["result_metric"]:
                pred.append(1)
                if label == 0:
                    FP += 1
                else:
                    TP += 1
            else:
                pred.append(0)
                if label == 0:
                    TN += 1
                else:
                    FN += 1
            # pred.append(score)
            true.append(label)


    eval_file.close()
    accuracy = metrics.accuracy_score(true, pred)
    # precision = TP / (TP + FP)
    # recall = TP / (TP + FN)
    # f = (2 * precision * recall) / (precision + recall)
    precision_macro = metrics.precision_score(true, pred, average='macro')
    # precision_micro = metrics.precision_score(true, pred, average='micro')
    recall_macro = metrics.recall_score(true, pred, average='macro')
    # recall_micro = metrics.recall_score(true, pred, average='micro')
    f_macro = metrics.f1_score(true, pred, average='macro')
    # f_micro = metrics.f1_score(true, pred, average='micro')
    # return accuracy, precision_macro, precision_micro, recall_macro, recall_micro, f_macro, f_micro
    # return accuracy, precision, recall, f, precision_macro, recall_macro, f_macro
    return accuracy, precision_macro, recall_macro, f_macro




