from sklearn import metrics


def evaluate(predictions, config):
    '''
    :param predictions: file_key = example["pre_topic"] + "_" + example["post_topic"]
    predictions[file_key] = [final_score, original_label, example["pre_topic"], example["post_topic"]]
    :param config:
    :return: precision, recall, f
    '''

    pred = []   #预测结果
    true = []   #实际标签
    with open(config["eval_result_file"], 'w') as eval_file:
        for file_key, result in predictions.items():
            score = result[0]
            label = result[1]
            pre_topic = result[2]
            post_topic = result[3]
            #print("pre topic: " + pre_topic + '\t' + "post topic: " + post_topic + "\t" + "predicted score: " + str(score) + '\t' + "true label: " + str(label))
            eval_file.write("pre topic: " + pre_topic + '\t' + "post topic: " + post_topic + "\t" + "predicted score: " + str(score) + '\t' + "true label: " + str(label) + '\n')
            if score > config["result_metric"]:
                pred.append(1)
            else:
                pred.append(0)
            # pred.append(score)
            true.append(label)
    eval_file.close()
    accuracy = metrics.accuracy_score(true, pred)
    precision_macro = metrics.precision_score(true, pred, average='macro')
    precision_micro = metrics.precision_score(true, pred, average='micro')
    recall_macro = metrics.recall_score(true, pred, average='macro')
    recall_micro = metrics.recall_score(true, pred, average='micro')
    f = metrics.f1_score(true, pred, average='macro')
    return accuracy, precision_macro, precision_micro, recall_macro, recall_micro, f




