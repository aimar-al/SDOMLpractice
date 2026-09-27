import os
import sys
import random
import torch
import pandas as pd
import numpy as np
from sklearn import metrics
import gradio as gr

# Add src to python path
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from dataset import CSVDataModule, CSVDataset
import plots
from modeling.train import Net, train
from modeling.predict import predict, predict_single

# Paths
DEFAULT_DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "benchmark_ai_detection_multimodel_2026.csv")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
LOG_DIR = os.path.join(os.path.dirname(__file__), "logs")
CHECKPOINT_PATH = os.path.join(MODEL_DIR, "best.ckpt")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# Load dataset and default checkpoint
dataset_instance = CSVDataset(csv_path=DEFAULT_DATA_PATH)
df_data = dataset_instance.df

current_net = None
if os.path.exists(CHECKPOINT_PATH):
    try:
        current_net = Net.load_from_checkpoint(CHECKPOINT_PATH)
    except Exception:
        current_net = None

# ----------------- Tab 1: Data Exploration -----------------
def get_exploration_plot(plot_type):
    """Generates the requested data exploration figure."""
    if plot_type == "Class Distribution (Human vs AI)":
        return plots.plot_class_distribution(df_data)
    elif plot_type == "Distribution by Text Domain":
        return plots.plot_domain_distribution(df_data)
    elif plot_type == "Perplexity Distribution by Class":
        return plots.plot_perplexity_distribution(df_data)
    elif plot_type == "Readability Score Distribution":
        return plots.plot_readability_distribution(df_data)
    elif plot_type == "Document Length vs Sentence Length":
        return plots.plot_length_scatterplot(df_data)
    elif plot_type == "Pearson Correlation Matrix":
        return plots.plot_correlation_matrix(df_data)
    elif plot_type == "Perplexity by Source Model":
        return plots.plot_source_model_distribution(df_data)
    return plots.plot_class_distribution(df_data)

# ----------------- Tab 2: Training & Evaluation -----------------
def run_training_and_eval(epochs, batch_size, learning_rate):
    """Trains the model with specified hyperparameters and evaluates performance."""
    global current_net
    data_module = CSVDataModule(data_dir=DEFAULT_DATA_PATH, batch_size=int(batch_size))
    data_module.setup()
    
    trainer = train(
        data_module,
        model_path=MODEL_DIR,
        logger_path=LOG_DIR,
        max_epochs=int(epochs),
        lr=float(learning_rate)
    )
    current_net = trainer.lightning_module
    
    # Read latest training log for loss and accuracy curves
    train_log_path = os.path.join(LOG_DIR, "train_log")
    curve_fig = None
    if os.path.exists(train_log_path):
        version_dirs = [d for d in os.listdir(train_log_path) if d.startswith("version_")]
        if version_dirs:
            latest_version = sorted(version_dirs, key=lambda x: int(x.split("_")[1]))[-1]
            metrics_csv = os.path.join(train_log_path, latest_version, "metrics.csv")
            if os.path.exists(metrics_csv):
                curve_fig = plots.loss_accuracy_curves(log_path=metrics_csv)
    
    # Model evaluation on the dataset
    y_true = data_module.train_ds.y
    last_ckpt = os.path.join(MODEL_DIR, "last.ckpt")
    eval_ckpt = last_ckpt if os.path.exists(last_ckpt) else CHECKPOINT_PATH
    y_pred, y_prob = predict(trainer, data_module, eval_ckpt)
    
    y_pred_np = y_pred.numpy()
    y_prob_np = y_prob.numpy()
    
    acc = metrics.accuracy_score(y_true, y_pred_np)
    prec = metrics.precision_score(y_true, y_pred_np, zero_division=0)
    rec = metrics.recall_score(y_true, y_pred_np, zero_division=0)
    f1 = metrics.f1_score(y_true, y_pred_np, zero_division=0)
    
    metrics_summary = pd.DataFrame([{
        "Accuracy": f"{acc:.4f}",
        "Precision": f"{prec:.4f}",
        "Recall": f"{rec:.4f}",
        "F1-Score": f"{f1:.4f}"
    }])
    
    cm_fig = plots.confusion_matrix(y_true, y_pred_np)
    roc_fig = plots.roc_curve(y_true, y_prob_np)
    pr_fig = plots.precision_recall_curve(y_true, y_prob_np)
    cal_fig = plots.calibration_curve(y_true, y_prob_np)
    
    status_msg = f"Training completed successfully ({epochs} epochs, batch size={batch_size}, lr={learning_rate})."
    return status_msg, curve_fig, metrics_summary, cm_fig, roc_fig, pr_fig, cal_fig

# ----------------- Tab 3: Interactive Benchmark -----------------
def load_random_text():
    """Selects a random sample from the dataset and formats its metadata."""
    idx = random.randint(0, len(df_data) - 1)
    row = df_data.iloc[idx]
    
    text_info = (
        f"**Domain:** {row['domain_context']}  \n"
        f"**Word Count:** {row['word_count']} | "
        f"**Avg Sentence Length:** {row['avg_sentence_length']:.2f}  \n"
        f"**Readability Score:** {row['readability_score']:.2f} | "
        f"**Simulated Perplexity:** {row['simulated_perplexity']:.2f}"
    )
    return row['text_content'], text_info, idx, ""

def evaluate_user_and_model(user_choice, sample_idx, current_scores):
    """Compares user guess and model prediction against ground truth."""
    global current_net
    if sample_idx is None or sample_idx == "":
        return "Please load a text sample first using the 'Load Random Text' button.", current_scores
    if not user_choice:
        return "Please select your prediction (Human or AI Generated).", current_scores
    
    idx = int(sample_idx)
    row = df_data.iloc[idx]
    real_label = int(row['is_ai_generated'])
    real_label_str = "AI Generated" if real_label == 1 else "Human"
    source_model = row['source_model']
    
    # User result
    user_label = 1 if user_choice == "AI Generated" else 0
    user_correct = (user_label == real_label)
    
    # Model result
    if current_net is None and os.path.exists(CHECKPOINT_PATH):
        current_net = Net.load_from_checkpoint(CHECKPOINT_PATH)
        
    if current_net is not None:
        feat_vector = dataset_instance.X[idx]
        pred_class, probs = predict_single(current_net, feat_vector)
        model_label_str = "AI Generated" if pred_class == 1 else "Human"
        model_correct = (pred_class == real_label)
        confidence = probs[pred_class] * 100
        model_feedback = f"The model predicted: **{model_label_str}** ({confidence:.1f}% confidence) -> " + (
            "**[Correct] The model was right.**" if model_correct else "**[Incorrect] The model was wrong.**"
        )
    else:
        model_correct = False
        model_feedback = "Model not loaded. Please train the model in Tab 2 first."
        
    # Update scores
    scores = current_scores or {"user": 0, "model": 0, "total": 0}
    scores["total"] += 1
    if user_correct:
        scores["user"] += 1
    if model_correct:
        scores["model"] += 1
        
    user_feedback = "**Correct!**" if user_correct else "**Incorrect.**"
    
    result_markdown = f"""
### Results:
- **True Origin:** **{real_label_str}** *(Source: {source_model})*
- **Your Choice:** {user_choice} -> {user_feedback}
- **Model Prediction:** {model_feedback}

---
**Cumulative Score:**
- **Your Score:** {scores['user']} / {scores['total']} ({(scores['user']/scores['total'])*100:.1f}%)
- **Model Score:** {scores['model']} / {scores['total']} ({(scores['model']/scores['total'])*100:.1f}%)
"""
    return result_markdown, scores

# ----------------- Gradio UI Construction -----------------
with gr.Blocks(title="AI Generated Text Classifier") as demo:
    gr.Markdown("# AI Generated Text Classifier")
    gr.Markdown("Interactive demonstration featuring dataset exploration, PyTorch Lightning training, and evaluation benchmark.")
    
    with gr.Tabs():
        # Tab 1: Data Exploration
        with gr.Tab("1. Data Exploration"):
            gr.Markdown("### Dataset Visualizations and Exploratory Analysis")
            with gr.Row():
                with gr.Column(scale=1):
                    plot_dropdown = gr.Dropdown(
                        choices=[
                            "Class Distribution (Human vs AI)",
                            "Distribution by Text Domain",
                            "Perplexity Distribution by Class",
                            "Readability Score Distribution",
                            "Document Length vs Sentence Length",
                            "Pearson Correlation Matrix",
                            "Perplexity by Source Model"
                        ],
                        value="Class Distribution (Human vs AI)",
                        label="Select a visualization"
                    )
                    show_plot_btn = gr.Button("Show Plot")
                with gr.Column(scale=2):
                    exploration_plot = gr.Plot(value=get_exploration_plot("Class Distribution (Human vs AI)"))
            
            show_plot_btn.click(
                fn=get_exploration_plot,
                inputs=plot_dropdown,
                outputs=exploration_plot
            )
            plot_dropdown.change(
                fn=get_exploration_plot,
                inputs=plot_dropdown,
                outputs=exploration_plot
            )
            
            gr.Markdown("### Dataset Sample")
            with gr.Row():
                gr.Dataframe(value=df_data.head(8), label="First rows of the dataset")

        # Tab 2: Training & Evaluation
        with gr.Tab("2. Training & Evaluation"):
            gr.Markdown("### Hyperparameter Configuration")
            with gr.Row():
                epochs_slider = gr.Slider(minimum=1, maximum=30, value=20, step=1, label="Epochs")
                batch_dropdown = gr.Dropdown(choices=[16, 32, 64, 128], value=64, label="Batch Size")
                lr_slider = gr.Slider(minimum=0.0001, maximum=0.01, value=0.001, step=0.0005, label="Learning Rate")
            
            train_btn = gr.Button("Train Model", variant="primary")
            train_status = gr.Textbox(label="Status", value="Ready to train.")
            
            gr.Markdown("### Training Progression")
            train_curves = gr.Plot(label="Loss & Accuracy Curves")
            
            gr.Markdown("### Model Evaluation (Evaluated on Trained Model)")
            eval_metrics_table = gr.Dataframe(label="Overall Performance Metrics")
            
            with gr.Row():
                cm_plot = gr.Plot(label="Confusion Matrix")
                roc_plot = gr.Plot(label="ROC Curve")
            with gr.Row():
                pr_plot = gr.Plot(label="Precision-Recall Curve")
                cal_plot = gr.Plot(label="Calibration Curve")
            
            train_btn.click(
                fn=run_training_and_eval,
                inputs=[epochs_slider, batch_dropdown, lr_slider],
                outputs=[train_status, train_curves, eval_metrics_table, cm_plot, roc_plot, pr_plot, cal_plot]
            )

        # Tab 3: Interactive Challenge
        with gr.Tab("3. Human or AI? (Challenge)"):
            gr.Markdown("### Can you tell whether a text was written by a human or generated by an AI?")
            gr.Markdown("Review the text features and content, submit your guess, and see whether you and the model were correct.")
            
            score_state = gr.State({"user": 0, "model": 0, "total": 0})
            sample_index_state = gr.State(None)
            
            load_btn = gr.Button("Load Random Text", variant="secondary")
            
            with gr.Row():
                with gr.Column(scale=1):
                    info_display = gr.Markdown("*(Click 'Load Random Text' to begin)*")
                with gr.Column(scale=2):
                    text_display = gr.Textbox(label="Text Content", lines=6, interactive=False)
            
            user_radio = gr.Radio(choices=["Human", "AI Generated"], label="Your Guess:")
            check_btn = gr.Button("Check Answer", variant="primary")
            
            result_display = gr.Markdown()
            
            load_btn.click(
                fn=load_random_text,
                inputs=[],
                outputs=[text_display, info_display, sample_index_state, result_display]
            )
            
            check_btn.click(
                fn=evaluate_user_and_model,
                inputs=[user_radio, sample_index_state, score_state],
                outputs=[result_display, score_state]
            )

if __name__ == "__main__":
    demo.launch()
