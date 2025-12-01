import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import io

# Create oversized figure
fig, ax = plt.subplots(figsize=(8, 8), dpi=100)

sns.heatmap(correlation_matrix, annot=True, fmt='.2f', 
            cmap='coolwarm', center=0, ax=ax)

plt.tight_layout()

# Save to buffer
buf = io.BytesIO()
plt.savefig(buf, format='png', bbox_inches='tight', dpi=100)
buf.seek(0)
plt.close()

# Resize to exact 512x512
img = Image.open(buf)
img = img.resize((512, 512), Image.Resampling.LANCZOS)
img.save('heatmap.png')
