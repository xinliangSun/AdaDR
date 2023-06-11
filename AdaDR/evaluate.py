import torch as th
from sklearn import metrics


def evaluate(args, model, graph_data,
             drug_graph, drug_feat, drug_sim_feat,
             dis_graph, dis_feat, dis_sim_feat):
    # rating_values = dataset.test_truths
    rating_values = graph_data['test'][2]
    # enc_graph = dataset.test_enc_graph.int().to(args.device)
    # dec_graph = dataset.test_dec_graph.int().to(args.device)
    enc_graph = graph_data['test'][0].int().to(args.device)
    dec_graph = graph_data['test'][1].int().to(args.device)

    model.eval()
    with th.no_grad():
        pred_ratings, _, _, _, _ = model(enc_graph, dec_graph,
                                         drug_graph, drug_sim_feat, drug_feat,
                                         dis_graph, dis_sim_feat, dis_feat)

    y_score = pred_ratings.view(-1).cpu().tolist()
    y_true = rating_values.cpu().tolist()
    # auc = metrics.roc_auc_score(y_true, y_score)
    # aupr = metrics.average_precision_score(y_true, y_score)
    fpr, tpr, _ = metrics.roc_curve(y_true, y_score)
    auc = metrics.auc(fpr, tpr)

    precision, recall, _ = metrics.precision_recall_curve(y_true, y_score)
    aupr = metrics.auc(recall, precision)

    return auc, aupr, y_true, y_score
