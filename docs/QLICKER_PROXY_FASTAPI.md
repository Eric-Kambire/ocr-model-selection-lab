# Qlicker : proxy système, diagnostic et passerelle FastAPI

Ce document explique comment un poste interne joint l'API Qlicker lorsque
Postman fonctionne mais qu'un script Python échoue.

## Le chemin réel de la requête

```text
Navigateur local
    → Gradio ou FastAPI sur 127.0.0.1
        → processus Python sur le PC interne
            → proxy Windows éventuel
                → API Qlicker interne
```

`127.0.0.1` signifie que Gradio/FastAPI n'est accessible que depuis le poste
qui l'exécute. L'appel Qlicker ne passe pas par un service cloud : il est émis
par ce processus Python local.

## Postman : proxy système ou proxy personnalisé

Dans Postman, ouvrir **Settings → Proxy**.

- **Use system proxy** : Postman utilise la configuration Windows du poste.
- **Use custom proxy configuration** : Postman utilise son propre hôte/port.
  Recopier cet hôte/port dans la configuration de la passerelle, sans partager
  les éventuels identifiants.

Le fait que Postman fonctionne ne prouve pas encore que Python suit le même
chemin : Postman peut utiliser un proxy Windows, un PAC, des cookies, un
certificat client ou une configuration TLS différente.

## Lire le proxy Windows depuis PowerShell

Exécuter sur le **poste interne où le script sera lancé** :

```powershell
Get-ItemProperty "HKCU:\Software\Microsoft\Windows\CurrentVersion\Internet Settings" |
Select-Object ProxyEnable, ProxyServer, ProxyOverride, AutoConfigURL, AutoDetect
```

Interprétation :

| Valeur | Signification | Effet pour le code |
|---|---|---|
| `ProxyEnable = 1` et `ProxyServer` rempli | Proxy manuel Windows | La passerelle FastAPI le lit et le transmet à HTTPX. |
| `ProxyOverride` rempli | Exceptions proxy | Les hôtes indiqués passent directement. |
| `AutoConfigURL` rempli | Script PAC | Postman peut le résoudre ; HTTPX ne l'exécute pas directement. |
| `AutoDetect = 1` | Détection automatique/WPAD | À confirmer avec l'équipe réseau si le proxy résolu est nécessaire. |

Vérifier également les variables d'environnement :

```powershell
Get-ChildItem Env: |
Where-Object Name -match "^(HTTP_PROXY|HTTPS_PROXY|NO_PROXY|ALL_PROXY)$"
```

Et la configuration WinHTTP, qui peut être différente de celle de Postman :

```powershell
netsh winhttp show proxy
```

Ne copiez jamais dans un ticket, un commit ou un chat une URL de proxy contenant
un mot de passe.

## Laboratoire Gradio de diagnostic

Le script [qlicker_url_parser_lab.py](../scripts/qlicker_url_parser_lab.py)
permet de coller une URL, modifier ses paramètres et faire un GET. Il affiche :

- DNS et adresses IP résolues ;
- accès TCP direct par IP ;
- négociation TLS directe ;
- proxy Windows/Python détecté ;
- durée réelle, timeouts configurés et chaîne d'exceptions.

Lancement :

```powershell
python scripts/qlicker_url_parser_lab.py
```

Puis ouvrir <http://127.0.0.1:8112>.

Un `ConnectTimeout` à 21 secondes alors que le timeout de connexion vaut 30
secondes est possible : 30 secondes est un **maximum**, pas un délai imposé.
Un proxy, pare-feu, serveur ou VPN peut couper la tentative plus tôt.

## Passerelle FastAPI + HTTPX

[qlicker_fastapi_gateway.py](../scripts/qlicker_fastapi_gateway.py) est une
alternative séparée de Gradio. Elle utilise `httpx.AsyncClient`, et **n'utilise
pas `requests`** pour les appels sortants Qlicker.

FastAPI est le serveur local ; HTTPX est le client HTTP qui ouvre la connexion
vers Qlicker. Remplacer `requests` par FastAPI seul n'est pas possible : un
client HTTP reste toujours nécessaire.

### Installer et lancer

```powershell
pip install -r requirements.txt

# Remplacer par les hôtes réels autorisés de Qlicker.
$env:QLICKER_ALLOWED_HOSTS = "qlicker.intra.local,10.20.30.40"

python scripts/qlicker_fastapi_gateway.py
```

Ouvrir ensuite :

- <http://127.0.0.1:8120/health> : état local, liste blanche et proxy ;
- <http://127.0.0.1:8120/docs> : interface Swagger de test.

La passerelle écoute sur `127.0.0.1` par défaut. Elle ne doit pas être exposée
sur le réseau sans authentification, journalisation et validation de sécurité.

### Ordre d'utilisation

1. Appeler `POST /v1/qlicker/parse-url` avec l'URL complète fournie par Qlicker.
   La réponse contient `endpoint` et la liste des paramètres, y compris les
   valeurs vides et paramètres répétés.
2. Modifier les paramètres dans Swagger ou dans un futur client Gradio.
3. Appeler `POST /v1/qlicker/get`.

Exemple de corps pour le GET :

```json
{
  "endpoint": "https://qlicker.intra.local/api/GetCustomers",
  "parameters": [
    {"name": "page", "value": "1"},
    {"name": "pageSize", "value": "20"}
  ],
  "connect_timeout_seconds": 30,
  "read_timeout_seconds": 300,
  "use_system_proxy": true
}
```

La liste blanche `QLICKER_ALLOWED_HOSTS` est obligatoire. Elle empêche que la
passerelle devienne un proxy libre vers n'importe quelle URL (risque SSRF).

## Lire une erreur correctement

| Erreur | Signification pratique |
|---|---|
| `ConnectTimeout` | La connexion au serveur ou au proxy n'a pas été établie avant le délai maximal. |
| `ProxyError` | Le proxy est injoignable, refuse la négociation ou demande une authentification. |
| `ReadTimeout` | La connexion existe, mais aucune réponse utile n'arrive assez vite. |
| `SSLError` | TLS/certificat interne/inspection HTTPS à corriger ; ne pas désactiver la vérification du certificat. |
| HTTP `401` / `403` | Le réseau fonctionne ; l'API attend une authentification, un cookie ou une autorisation. |
| HTTP `4xx` / `5xx` | Le serveur a répondu ; comparer alors URL, paramètres et headers avec Postman. |

Pour un incident, conserver : type d'erreur, durée réelle, hôte/port, mode
proxy et chaîne d'exceptions. Ne pas conserver les données client, les tokens
ou mots de passe.
