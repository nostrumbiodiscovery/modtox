import os

# Dataframe column names
MOL_NAME_COLUMN = "Title"
ACTIVITY_COLUMN = 'Activity'

# Input filenames
ACTIVES_SDF = 'actives.sdf'
INACTIVES_SDF = 'inactives.sdf'
GLIDE_FEATURES = 'glide_features.csv'

# Output filenames
BALANCED_GLIDE = 'balanced_glide.csv'

# Saving directory
SAVEDIR = os.getcwd()

# Set proportion (internal test and external test)
INTERNAL_PROPORTION = 0.3
EXTERNAL_PROPORTION = 0.05
