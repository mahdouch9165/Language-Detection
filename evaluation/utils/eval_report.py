import json

import sklearn
import numpy as np
import matplotlib.pyplot as plt

import global_id_utils

def make_evaluation_outputs(prediction_output, output_dir, model_id_to_global_id, model_id2label, dataset_id_to_global_id):
    """
    Runs all evaluations on the model's predictions, saving output in output_dir.
    Evaluations are:
        - Classification report for accuracy, recacll, f1 across all true classes
        - Accuracy bar plot across true classes
        - Confusion Matrices (Counts, normalized over true)
    """
    prediction_model_ids = np.argmax(prediction_output.predictions, axis=-1)
    label_model_ids = prediction_output.label_ids

    prediction_global_ids = [model_id_to_global_id[id] for id in prediction_model_ids]
    label_global_ids = [model_id_to_global_id[id] for id in label_model_ids]

    def model_id_to_global_name(model_id):
        global_id = model_id_to_global_id[model_id]
        lang = global_id_utils.global_id_to_lang(global_id)
        name = lang.name
        return name

    unique_prediction_model_ids, unique_prediction_names = unique_ids_and_names_func(prediction_model_ids, model_id_to_global_name)
    unique_label_model_ids, unique_label_names = unique_ids_and_names_func(label_model_ids, model_id_to_global_name)

    _, report_text = save_report(output_dir, prediction_model_ids, label_model_ids, unique_label_model_ids, unique_label_names)
    print(report_text)

    all_unique_ids = np.unique(np.concatenate((label_model_ids, prediction_model_ids)))
    all_unique_names = list(map(model_id_to_global_name, all_unique_ids))

    create_and_save_visualizations(output_dir, prediction_model_ids, label_model_ids, all_unique_ids, all_unique_names)

    save_predictions(output_dir, prediction_model_ids, label_model_ids, prediction_global_ids, label_global_ids)

def unique_ids_and_names(ids: list[int], id_to_name):
    """
    Args:
        - ids: list/array of ids
        - id_to_name: Mapping of integer ids to string names
    """
    unique_ids = np.unique(ids)
    unique_names = [id_to_name[id] for id in unique_ids]
    
    return unique_ids, unique_names

def unique_ids_and_names_func(ids: list[int], id_to_name):
    """
    Args:
        - ids: list/array of ids
        - id_to_name: Function mapping of integer ids to string names
    """
    unique_ids = np.unique(ids)
    unique_names = list(map(id_to_name, unique_ids))
    
    return unique_ids, unique_names


def create_and_save_visualizations(output_dir:str, prediction_model_ids, label_model_ids, all_unique_ids, all_unique_names):
    #################
    # Confusion matrix of counts
    #################

    cm = sklearn.metrics.confusion_matrix(
        y_true=label_model_ids,
        y_pred=prediction_model_ids,
        labels=all_unique_ids,
    )
    disp = sklearn.metrics.ConfusionMatrixDisplay(confusion_matrix=cm,
                                                  display_labels=all_unique_names)

    counts_cm_fig, counts_cm_ax = plt.subplots(figsize=(10,10))

    disp.plot(ax=counts_cm_ax)
    counts_cm_ax.set_title("Confusion Matrix (counts)")
    counts_cm_ax.tick_params(axis="x", labelrotation=90)

    # Prevent labels from getting cut off when saving image
    counts_cm_fig.tight_layout()
    counts_cm_fig.savefig(output_dir + "/confusion matrix.png")
    counts_cm_fig.savefig(output_dir + "/confusion matrix.svg")
    counts_cm_fig.show()
    
    # Accuracy of languages evaluated (only languages in the true labels)
    accuracy = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
    accuracy = accuracy.diagonal()

    not_nans = ~np.isnan(accuracy)
    accuracy = accuracy[not_nans]
    labels = np.array(all_unique_names)[not_nans]

    accuracy_fig, accuracy_ax = plt.subplots()

    accuracy_ax.bar(np.arange(len(accuracy)), accuracy, label=labels)
    accuracy_ax.set_title("Accuracy per language")
    accuracy_ax.set_ylabel("Accuracy")
    accuracy_ax.set_xlabel("Language")
    accuracy_ax.set_yticks(np.linspace(0, 1, num=6))
    accuracy_ax.set_xticks(np.arange(len(accuracy)), labels, rotation=45, horizontalalignment="right")
    # Prevent labels from getting cut off when saving image
    accuracy_fig.tight_layout()

    accuracy_fig.savefig(output_dir + "/accuracy bar plot.png")
    accuracy_fig.savefig(output_dir + "/accuracy bar plot.svg")
    accuracy_fig.show()
    
    #################
    # Confusion matrix normalized over true labels
    #################

    cm = sklearn.metrics.confusion_matrix(
        y_true=label_model_ids,
        y_pred=prediction_model_ids,
        labels=all_unique_ids,
        normalize="true",
    )
    disp = sklearn.metrics.ConfusionMatrixDisplay(confusion_matrix=cm,
                                                  display_labels=all_unique_names)

    fig, ax = plt.subplots(figsize=(15,15))
    ax.set_title("Confusion matrix (normalized over true rows)")

    disp.plot(ax=ax)
    # ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment="right")
    ax.tick_params(axis="x", labelrotation=90)
    # Prevent labels from getting cut off when saving image
    fig.tight_layout()

    fig.savefig(output_dir + "/confusion matrix normalized over true.png")
    fig.savefig(output_dir + "/confusion matrix normalized over true.svg")
    fig.show()

    fig, ax = plt.subplots(figsize=(15,15))
    ax.set_title("Confusion matrix normalized over true labels")
    
    # Confusion matrix normalized over true labels (small font)

    sklearn.metrics.ConfusionMatrixDisplay.from_predictions(
        y_true=label_model_ids,
        y_pred=prediction_model_ids,
        labels=all_unique_ids,
        display_labels=all_unique_names,
        normalize="true",
        include_values=True,
        xticks_rotation="vertical",
        ax=ax,
        text_kw={"size": 6},
        values_format=".2g",
    )
    # Prevent labels from getting cut off when saving image
    fig.tight_layout()
    fig.savefig(output_dir + "/confusion matrix normalized over true small font.png")
    fig.savefig(output_dir + "/confusion matrix normalized over true small font.svg")
    fig.show()
    
def save_report(output_dir: str, prediction_model_ids, label_model_ids, unique_label_ids, unique_label_names):
    classification_report_args = dict(
        y_true=label_model_ids,
        y_pred=prediction_model_ids,
        labels=unique_label_ids,
        target_names=unique_label_names,
        zero_division=0,
    )
    
    classification_report = sklearn.metrics.classification_report(**classification_report_args, output_dict=True)
    
    with open(output_dir + "/" + "classification_report.json", "w") as out_file:
        json.dump(classification_report, out_file)
    classification_report_text = sklearn.metrics.classification_report(**classification_report_args, output_dict=False)
    with open(output_dir + "/" + "classification_report.txt", "w") as out_file:
        out_file.write(classification_report_text)

    return classification_report, classification_report_text

def save_predictions(output_dir: str, prediction_model_ids, label_model_ids, prediction_global_ids, label_global_ids):
    results = {
        "predictions": prediction_global_ids,
        "labels": label_global_ids
    }
    with open(output_dir + "/predictions.json", "w") as out_file:
        json.dump(results, out_file)

    predictions_readable = [global_id_utils.global_id_to_iso639_part3(id) for id in prediction_global_ids]
    labels_readable = [global_id_utils.global_id_to_iso639_part3(id) for id in label_global_ids]
    results_readable = {
        "predictions": predictions_readable,
        "labels": labels_readable
    }
    with open(output_dir + "/predictions_readable.json", "w") as out_file:
        json.dump(results, out_file)
