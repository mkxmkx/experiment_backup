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
            if score > 0.3:
                pred.append(1)
            else:
                pred.append(0)
            # pred.append(score)
            true.append(label)
    eval_file.close()
    precision = metrics.precision_score(true, pred, average='macro')
    recall = metrics.recall_score(true, pred, average='macro')
    f = metrics.f1_score(true, pred, average='weighted')
    return precision, recall, f




