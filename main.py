from master import MASTERModel
import pickle
import numpy as np
import time
import pandas as pd
import os
import json
from datetime import datetime
import torch
import logging

# --- Logging Setup ---
log_format = "%(asctime)s - %(levelname)s - %(message)s"
logging.basicConfig(
    level=logging.INFO,
    format=log_format,
    handlers=[
        logging.StreamHandler(),  # Log to console
        logging.FileHandler("training.log"),  # Log to a general file
    ],
)
# --- End Logging Setup ---

# Create results directory if it doesn't exist
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

# Please install qlib first before load the data.
logging.info("Loading data...")
universe = "csi300"  # ['csi300','csi800']
prefix = "opensource"  # ['original','opensource'], which training data are you using
train_data_dir = f"data"
with open(f"{train_data_dir}\{prefix}\{universe}_dl_train.pkl", "rb") as f:
    dl_train = pickle.load(f)

predict_data_dir = f"data\opensource"
with open(f"{predict_data_dir}\{universe}_dl_valid.pkl", "rb") as f:
    dl_valid = pickle.load(f)
with open(f"{predict_data_dir}\{universe}_dl_test.pkl", "rb") as f:
    dl_test = pickle.load(f)

print("Data Loaded.")
logging.info("Data loaded successfully.")


# --- Model & Training Configuration ---
d_feat = 158
d_model = 256
t_nhead = 4
s_nhead = 2
dropout = 0.5
gate_input_start_index = 158
gate_input_end_index = 221

if universe == "csi300":
    beta = 5
elif universe == "csi800":
    beta = 2

n_epoch = 100
lr = 1e-5
GPU = 0
train_stop_loss_thred = 0.95
# --- End Configuration ---


# --- Helper Functions ---
def convert_to_json_serializable(obj):
    """Recursively convert NumPy types to native Python types for JSON."""
    if isinstance(obj, (np.integer, np.int64, np.int32, np.int16, np.int8)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    elif isinstance(obj, (np.bool_)):
        return bool(obj)
    elif isinstance(obj, dict):
        return {key: convert_to_json_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_json_serializable(item) for item in obj]
    return obj


def save_json_log(data, filepath):
    """Safely save data to JSON, converting NumPy types."""
    try:
        serializable_data = convert_to_json_serializable(data)
        with open(filepath, "w") as f:
            json.dump(serializable_data, f, indent=4)
        return True
    except Exception as e:
        logging.error(f"Failed to save JSON log to {filepath}: {e}")
        return False


# --- End Helper Functions ---


# --- Training Loop ---
ic_all_seeds = []
icir_all_seeds = []
ric_all_seeds = []
ricir_all_seeds = []

for seed in [0]:
    logging.info(f"--- Starting Training: Seed {seed} ---")
    # Generate timestamp and filenames
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename_base = f"MASTERModel_{universe}_{prefix}_{timestamp}_seed{seed}"
    json_log_filepath = os.path.join(RESULTS_DIR, f"{log_filename_base}.json")
    csv_results_filepath = os.path.join(RESULTS_DIR, f"{log_filename_base}.csv")

    # Initialize log structure
    log_data = {
        "run_info": {
            "start_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "model_class": "MASTERModel",
            "prefix": f"{universe}_{prefix}",
            "seed": seed,
            "device": f"cuda:{GPU}" if GPU is not None else "cpu",
            "log_file": json_log_filepath,
            "results_file": csv_results_filepath,
        },
        "hyperparameters": {
            "d_feat": d_feat,
            "d_model": d_model,
            "t_nhead": t_nhead,
            "s_nhead": s_nhead,
            "dropout": dropout,
            "beta": beta,
            "n_epochs": n_epoch,
            "learning_rate": lr,
            "early_stopping_threshold": train_stop_loss_thred,
        },
        "training_history": {"epochs": []},  # Removed redundant keys
        "status": "training_started",
    }

    # Save initial log
    save_json_log(log_data, json_log_filepath)

    # Instantiate Model
    model = MASTERModel(
        d_feat=d_feat,
        d_model=d_model,
        t_nhead=t_nhead,
        s_nhead=s_nhead,
        T_dropout_rate=dropout,
        S_dropout_rate=dropout,
        beta=beta,
        gate_input_end_index=gate_input_end_index,
        gate_input_start_index=gate_input_start_index,
        n_epochs=n_epoch,
        lr=lr,
        GPU=GPU,
        seed=seed,
        train_stop_loss_thred=train_stop_loss_thred,
        save_path="model",  # Original save path
        save_prefix=f"{universe}_{prefix}",  # Original save prefix
    )

    start_time = time.time()
    best_model_state = None
    best_val_ic = -np.inf  # Initialize with negative infinity

    # --- Custom Fit Function with Logging ---
    original_fit = model.fit

    def log_epoch_results(epoch, train_loss, val_metrics, is_best):
        """Append epoch results to the JSON log file."""
        # nonlocal log_data # Ensure we modify the outer scope log_data - nonlocal is not needed here
        epoch_log = {
            "epoch": epoch,
            "train_loss": train_loss,
            "validation_metrics": val_metrics,
            "is_best_epoch": is_best,
        }
        # Add saved model info only if applicable (handled during saving)

        try:
            # It's safer to reload the log file in case of concurrent runs (though unlikely here)
            # However, for simplicity in this single loop, we update the in-memory dict
            # Read the current log data first to append
            current_log_data = {}
            if os.path.exists(json_log_filepath):
                with open(json_log_filepath, "r") as f:
                    try:
                        current_log_data = json.load(f)
                    except json.JSONDecodeError:
                        logging.warning(
                            f"Could not decode existing JSON log file: {json_log_filepath}. Will overwrite."
                        )
                        current_log_data = (
                            log_data  # Fallback to initial if file corrupted
                        )
            else:
                current_log_data = log_data  # Use initial if file doesn't exist

            # Ensure training_history and epochs keys exist
            if "training_history" not in current_log_data:
                current_log_data["training_history"] = {"epochs": []}
            elif "epochs" not in current_log_data["training_history"]:
                current_log_data["training_history"]["epochs"] = []

            current_log_data["training_history"]["epochs"].append(epoch_log)
            if is_best:
                current_log_data["best_performance"] = {
                    "epoch": epoch,
                    "validation_IC": val_metrics["IC"],
                }
            save_json_log(current_log_data, json_log_filepath)
        except Exception as e:
            logging.warning(f"Could not update JSON log for epoch {epoch}: {e}")

    def logging_fit(dl_train_inner, dl_valid_inner=None):
        """Custom fit method that incorporates logging and best model saving."""
        # nonlocal best_model_state, best_val_ic, log_data # Access outer scope variables - nonlocal is not needed for modification of mutable types like dict
        global best_model_state, best_val_ic  # Use global or pass as args if preferred
        train_loader = model._init_data_loader(
            dl_train_inner, shuffle=True, drop_last=True
        )
        saved_best_model_this_run = False

        for epoch in range(model.n_epochs):
            model.model.train()  # Set model to training mode
            epoch_train_loss = model.train_epoch(train_loader)
            model.fitted = epoch  # Update fitted status

            epoch_val_metrics = {}
            is_best_epoch = False

            if dl_valid_inner:
                model.model.eval()  # Set model to evaluation mode
                # Use original predict which doesn't modify state unnecessarily
                # Ensure predict returns (predictions, metrics) or handle uncertainty if implemented
                val_result = model.predict(dl_valid_inner)
                if isinstance(val_result, tuple) and len(val_result) >= 2:
                    _, epoch_val_metrics = val_result[:2]  # Take first two elements
                else:
                    logging.error("Validation predict did not return expected tuple.")
                    continue  # Skip logging if prediction failed

                current_ic = epoch_val_metrics.get("IC", -np.inf)
                if current_ic > best_val_ic:
                    best_val_ic = current_ic
                    is_best_epoch = True
                    # Deep copy state dict on CPU
                    best_model_state = {
                        k: v.cpu().clone() for k, v in model.model.state_dict().items()
                    }
                    logging.info(
                        f"Epoch {epoch}: New best model found with validation IC: {best_val_ic:.4f}"
                    )

                # Log results for the epoch
                log_epoch_results(
                    epoch, epoch_train_loss, epoch_val_metrics, is_best_epoch
                )

                # Print progress
                metrics_str = ", ".join(
                    [f"{k}: {v:.4f}" for k, v in epoch_val_metrics.items()]
                )
                print(
                    f"Epoch {epoch}: Train Loss: {epoch_train_loss:.6f}, Validation Metrics: {{{metrics_str}}}"
                )
                logging.info(
                    f"Epoch {epoch}: Train Loss: {epoch_train_loss:.6f}, Validation Metrics: {{{metrics_str}}}"
                )
            else:
                # Log only train loss if no validation set
                log_epoch_results(epoch, epoch_train_loss, {}, False)
                print(f"Epoch {epoch}: Train Loss: {epoch_train_loss:.6f}")
                logging.info(f"Epoch {epoch}: Train Loss: {epoch_train_loss:.6f}")

            # Early Stopping Check (based on train loss as per original logic)
            if epoch_train_loss <= model.train_stop_loss_thred:
                logging.info(
                    f"Early stopping triggered at epoch {epoch} (Train Loss: {epoch_train_loss:.6f} <= {model.train_stop_loss_thred:.6f})"
                )
                # Save the best model found so far based on validation IC
                if best_model_state:
                    model_save_path = os.path.join(
                        model.save_path, f"{model.save_prefix}_{model.seed}.pkl"
                    )
                    torch.save(best_model_state, model_save_path)
                    logging.info(
                        f"Saved best model (Epoch {log_data.get('best_performance',{}).get('epoch','N/A')}) to {model_save_path}"
                    )
                    saved_best_model_this_run = True
                    log_data["saved_model_path"] = model_save_path  # Log saved path
                else:
                    logging.warning(
                        "Early stopping triggered, but no best model state saved (validation IC likely didn't improve)."
                    )
                break  # Exit training loop

        # After loop: Save the best model if early stopping didn't save it
        if not saved_best_model_this_run and best_model_state:
            model_save_path = os.path.join(
                model.save_path, f"{model.save_prefix}_{model.seed}.pkl"
            )
            torch.save(best_model_state, model_save_path)
            logging.info(
                f"Training finished. Saved best model (Epoch {log_data.get('best_performance',{}).get('epoch','N/A')}) to {model_save_path}"
            )
            log_data["saved_model_path"] = model_save_path  # Log saved path
        elif not best_model_state:
            logging.warning(
                "Training finished, but no best model state was ever saved (validation IC likely didn't improve or no validation data)."
            )

    # Replace the original fit method
    model.fit = logging_fit
    # --- End Custom Fit Function ---

    # --- Execute Training and Testing ---
    try:
        # Train the model using the custom fit function
        model.fit(dl_train, dl_valid)
        logging.info(f"Model training completed for seed {seed}.")
        print("Model Trained.")

        # Test the final model (or best saved one if loaded)
        # If the best model was saved, we might want to load it here before predicting
        # For simplicity, using the model state at the end of training/early stopping
        logging.info("Predicting on test data...")
        model.model.eval()  # Ensure model is in eval mode for testing
        test_results = model.predict(dl_test)

        # Handle potential uncertainty output
        if isinstance(test_results, tuple) and len(test_results) == 3:
            predictions, test_metrics, uncertainty = test_results
            results_df = pd.DataFrame(
                {"prediction": predictions, "uncertainty": uncertainty},
                index=dl_test.get_index(),  # Use index from dataloader
            )
            logging.info(
                f"Test predictions generated with uncertainty. Average uncertainty: {uncertainty.mean():.4f}"
            )
        elif isinstance(test_results, tuple) and len(test_results) == 2:
            predictions, test_metrics = test_results
            results_df = pd.DataFrame(
                {"prediction": predictions}, index=dl_test.get_index()
            )
            logging.info("Test predictions generated (no uncertainty).")
        else:
            logging.error("Test predict did not return expected tuple.")
            test_metrics = {}
            results_df = pd.DataFrame()  # Empty dataframe

        # Save test predictions to CSV
        if not results_df.empty:
            try:
                results_df.to_csv(csv_results_filepath)
                logging.info(f"Test results saved to {csv_results_filepath}")
            except Exception as e:
                logging.error(
                    f"Failed to save test results CSV to {csv_results_filepath}: {e}"
                )

        running_time = time.time() - start_time

        # Update JSON log with completion status and test metrics
        log_data["status"] = "completed"
        log_data["training_completed"] = {
            "end_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "total_epochs_trained": model.fitted + 1,
            "running_time_seconds": running_time,
        }
        log_data["test_metrics"] = test_metrics
        save_json_log(log_data, json_log_filepath)

        # Print final summary for the seed
        print(f"Seed {seed}: Training Time: {running_time:.2f} sec")
        print(f"Test Metrics: {test_metrics}")
        logging.info(f"Seed {seed}: Training Time: {running_time:.2f} sec")
        logging.info(f"Test Metrics: {test_metrics}")

        # Append metrics for overall calculation
        if test_metrics:
            ic_all_seeds.append(test_metrics.get("IC", np.nan))
            icir_all_seeds.append(test_metrics.get("ICIR", np.nan))
            ric_all_seeds.append(test_metrics.get("RIC", np.nan))
            ricir_all_seeds.append(test_metrics.get("RICIR", np.nan))

    except Exception as e:
        logging.error(
            f"An error occurred during training/testing for seed {seed}: {e}",
            exc_info=True,
        )
        # Update JSON log with error status
        log_data["status"] = "failed"
        log_data["error"] = str(e)
        log_data["error_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        save_json_log(log_data, json_log_filepath)
        # Optionally re-raise or handle as needed
        # raise e
    finally:
        # Clean up if necessary (e.g., clear GPU memory)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logging.info(f"--- Finished Training: Seed {seed} ---")
# --- End Training Loop ---

# --- Final Results Aggregation ---
if ic_all_seeds:  # Only calculate if list is not empty
    logging.info("--- Overall Results --- ")
    print("--- Overall Results --- ")
    print(f"IC: {np.nanmean(ic_all_seeds):.4f} pm {np.nanstd(ic_all_seeds):.4f}")
    print(f"ICIR: {np.nanmean(icir_all_seeds):.4f} pm {np.nanstd(icir_all_seeds):.4f}")
    print(f"RIC: {np.nanmean(ric_all_seeds):.4f} pm {np.nanstd(ric_all_seeds):.4f}")
    print(
        f"RICIR: {np.nanmean(ricir_all_seeds):.4f} pm {np.nanstd(ricir_all_seeds):.4f}"
    )
    logging.info(f"IC: {np.nanmean(ic_all_seeds):.4f} pm {np.nanstd(ic_all_seeds):.4f}")
    logging.info(
        f"ICIR: {np.nanmean(icir_all_seeds):.4f} pm {np.nanstd(icir_all_seeds):.4f}"
    )
    logging.info(
        f"RIC: {np.nanmean(ric_all_seeds):.4f} pm {np.nanstd(ric_all_seeds):.4f}"
    )
    logging.info(
        f"RICIR: {np.nanmean(ricir_all_seeds):.4f} pm {np.nanstd(ricir_all_seeds):.4f}"
    )
else:
    logging.warning("No results were successfully generated across seeds.")
# --- End Final Results Aggregation ---
