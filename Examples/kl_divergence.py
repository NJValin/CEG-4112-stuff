import tensorflow as tf
import tensorflow_probability as tfp
import matplotlib.pyplot as plt

tf_dist = tfp.distributions

# Define two distributions
distribution1 = tf_dist.Normal(loc=0.0, scale=1.0)# gaussian with mean=0 & std=1
distribution2 = tf_dist.Normal(loc=1.0, scale=2.0)# gaussian with mean=1 & std=2

# Calculate the KL divergence
kl_divergence = tf_dist.kl_divergence(distribution1, distribution2)

print(f'KL divergence: {kl_divergence}')

samples1 = distribution1.sample(10_000)
samples2 = distribution2.sample(10_000)
log_prob1 = distribution1.log_prob(samples1)
log_prob2 = distribution2.log_prob(samples2)

monte_carlo_kl = tf.reduce_mean(log_prob1-log_prob2)
print(f"Monte Carlo KL Divergence: {monte_carlo_kl}")

plt.figure(figsize=(10, 6))
plt.hist(samples1, bins=50, alpha=0.5, label='samples_dist1', density=True)
plt.hist(samples2, bins=50, alpha=0.5, label='samples_dist2', density=True)
plt.xlabel('Log Probability')
plt.ylabel('Density')
plt.title('Distribution of dist1 and dist2 samples')
plt.legend()
plt.show()
