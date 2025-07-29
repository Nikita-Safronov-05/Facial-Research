import pandas as pd
import numpy as np

pd.set_option('display.max_columns', None)
metadata = 'encoder_data/experiment_metadata.csv'
df = pd.read_csv(metadata)

# Columns to use as the unique config key
group_cols = ['latent_dim', 'patience', 'encoder_layers', 'decoder_layers', 'activation', 'batch_size']

# Aggregation: mean and std for validation loss, mean for best epoch (add more columns as needed)
agg_df = df.groupby(group_cols).agg(
    avg_val_loss_mean = ('avg_best_val_loss', 'mean'),
    avg_val_loss_std = ('avg_best_val_loss', 'std'),
    n_seeds = ('seed', 'count'),
    result_files = ('result_file', list),
    seeds = ('seed', list)
).reset_index()
print(agg_df.shape, agg_df.head(20))

import matplotlib.pyplot as plt

# Sort by latent_dim for cleaner plotting
plot_df = agg_df.sort_values('latent_dim')

plt.figure(figsize=(8, 5))
plt.errorbar(
    plot_df['latent_dim'],
    plot_df['avg_val_loss_mean'],
    yerr=plot_df['avg_val_loss_std'],
    capsize=5,
    label='Validation Loss (mean ± std)'
)
plt.xlabel('Latent Dimension')
plt.ylabel('Validation Loss')
plt.title('Latent Dimension vs. Validation Loss')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
# latent_dim, patience, encoder_layers, decoder_layers, activation, batch_size = agg_df.sort_values('avg_val_loss_mean').iloc[0][group_cols]
# print("Best config across seeds:")
# mask = (
#     (df['latent_dim'] == latent_dim) &
#     (df['patience'] == patience) &
#     (df['encoder_layers'] == encoder_layers) &
#     (df['decoder_layers'] == decoder_layers) &
#     (df['activation'] == activation) &
#     (df['batch_size'] == batch_size)
# )
# best_config_rows = df[mask]
# print(best_config_rows)