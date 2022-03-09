# %%
from turtle import update
import dcarte


# %% [markdown]
# ### Define list of patient random id's

# %%
pids = ['Mhy2uUxJnCtsEZbToCDDEE',
        'EzjmQuxceJzLATdtyUbDn8',
        'XsfBVz6GP7XtvhY4CiAvuf',
        'JEvSjBYo6KvDKHCGUnDNxm',
        'Hpbmfij5E6hh19u1uKFXQT',
        'DArgytxDspe3jPUMMLhnvk',
        '8MzeY8ZePeGa6kFKJpXoMm',
        'BL3rTLVVUbXDyCxtWhSky6',
        'YJMET4inKfFcBERma2tQRi',
        '4h1dAuzg9rdrhyojwxUS26',
        'PtSJCv5bDWvZe3f1V7wRgQ',
        'TTTuJeUvnGhPJBznMtCiNj',
        '2zbyXzYNKPwiPtjaA2L64o',
        '3hY7Mp7u9YPo1xMARSxLhc',
        'U2dZSjjycMm5bRNvHcLrAr']

# %% [markdown]
# ### Make sure that you have the profile domain by running `dcarte.domains()`
# If you don't have it reffer to tutorial 4. derived datasets  
# %%
dcarte.domains()
update = True
# %%
Activity_Dailies = dcarte.load('Activity_Dailies','profile',update=update)
Activity_Dailies = Activity_Dailies.query('patient_id in @pids')
Activity_Dailies.info()
# %%
Physiology_Dailies = dcarte.load('Physiology_Dailies','profile',update=update)
Physiology_Dailies = Physiology_Dailies.query('patient_id in @pids')
Physiology_Dailies.info()
# %%
Sleep_Dailies = dcarte.load('Sleep_Dailies','profile',update=update)
Sleep_Dailies = Sleep_Dailies.query('patient_id in @pids')
Sleep_Dailies.info()

# %%
Activity_Dailies.to_parquet('Activity_Dailies.parquet')
Physiology_Dailies.to_parquet('Physiology_Dailies.parquet')
Sleep_Dailies.to_parquet('Sleep_Dailies.parquet')
# %%
