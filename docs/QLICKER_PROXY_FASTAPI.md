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

## Procédure PowerShell pas à pas : identifier le proxy du poste

Ouvrir **PowerShell** avec le même compte Windows qui ouvre Postman et qui
lancera Python. Les valeurs sont souvent différentes pour un autre compte.

### Étape 1 — confirmer le compte Windows utilisé

```powershell
whoami
```

Sortie attendue, exemple :

```text
ENTREPRISE\prenom.nom
```

Si ce compte n'est pas celui qui utilise Postman, arrêter ici et ouvrir une
session PowerShell avec le bon compte.

### Étape 2 — lire la configuration proxy Windows (WinINET)

Copier ce bloc complet :

```powershell
$registryPath = "HKCU:\Software\Microsoft\Windows\CurrentVersion\Internet Settings"
$settings = Get-ItemProperty -Path $registryPath

[PSCustomObject]@{
  ProxyEnable   = $settings.ProxyEnable
  ProxyServer   = $settings.ProxyServer
  ProxyOverride = $settings.ProxyOverride
  AutoConfigURL = $settings.AutoConfigURL
  AutoDetect    = $settings.AutoDetect
} | Format-List
```

Sorties possibles et leur signification :

```text
# Cas A : proxy manuel
ProxyEnable   : 1
ProxyServer   : proxy.entreprise.local:8080
ProxyOverride : localhost;*.intra.local
AutoConfigURL :
AutoDetect    : 0
```

Le proxy à utiliser est `http://proxy.entreprise.local:8080`. `ProxyOverride`
liste les hôtes qui doivent rester en connexion directe.

```text
# Cas B : script PAC
ProxyEnable   : 0
ProxyServer   :
ProxyOverride :
AutoConfigURL : https://proxy.entreprise.local/config/proxy.pac
AutoDetect    : 0
```

Postman peut évaluer ce script PAC. Le script indique ensuite le proxy selon
l'URL demandée ; il n'y a donc pas nécessairement une seule adresse proxy.

```text
# Cas C : détection WPAD
ProxyEnable   : 0
ProxyServer   :
ProxyOverride :
AutoConfigURL :
AutoDetect    : 1
```

Windows cherche automatiquement une configuration proxy. L'adresse finale ne
peut pas être déduite seulement depuis ce tableau : demander le proxy résolu à
l'équipe réseau ou obtenir le PAC/WPAD.

```text
# Cas D : accès direct
ProxyEnable   : 0
ProxyServer   :
ProxyOverride :
AutoConfigURL :
AutoDetect    : 0
```

Il n'y a pas de proxy Windows visible. Si Postman fonctionne, il peut joindre
Qlicker directement via Cisco Secure Client/VPN ou posséder un proxy personnalisé.

> `ProxyServer` absent ou vide n'est pas une erreur : cela exclut simplement le
> cas « proxy manuel ».

### Étape 3 — vérifier les variables proxy du processus Python/curl

```powershell
Get-ChildItem Env: |
  Where-Object Name -match "^(HTTP_PROXY|HTTPS_PROXY|NO_PROXY|ALL_PROXY)$" |
  Format-Table Name, Value -AutoSize
```

Sortie possible :

```text
Name        Value
----        -----
HTTPS_PROXY http://proxy.entreprise.local:8080
NO_PROXY    localhost,127.0.0.1,.intra.local
```

Cela signifie que les programmes démarrés depuis cette fenêtre PowerShell
peuvent utiliser ce proxy. **Aucune sortie** signifie qu'aucune variable proxy
n'est définie ; ce n'est pas un échec.

### Étape 4 — comparer avec WinHTTP

```powershell
netsh winhttp show proxy
```

Sorties possibles :

```text
Current WinHTTP proxy settings:

    Proxy Server(s) :  proxy.entreprise.local:8080
    Bypass List     :  localhost;*.intra.local
```

ou :

```text
Direct access (no proxy server).
```

WinHTTP est une autre pile Windows. Une sortie « Direct access » n'annule pas
un proxy Postman/WinINET configuré à l'étape 2.

### Étape 5 — si un PAC est configuré, afficher ses règles

Exécuter uniquement si `AutoConfigURL` n'est pas vide :

```powershell
$pacUrl = $settings.AutoConfigURL
$pacContent = (Invoke-WebRequest -Uri $pacUrl -UseBasicParsing).Content
$pacContent | Select-String -Pattern "PROXY|SOCKS|DIRECT" -AllMatches
```

Sortie possible :

```text
return "PROXY proxy.entreprise.local:8080; DIRECT";
```

Cette ligne signifie : essayer le proxy, puis la connexion directe si le proxy
échoue. Un PAC peut toutefois contenir beaucoup de conditions ; son contenu ne
prouve pas encore quelle règle est choisie pour l'URL Qlicker précise.

### Étape 6 — tester la route directe vers Qlicker

Remplacer `SERVEUR_QCLICKER` par le nom réel, sans ajouter de token :

```powershell
Test-NetConnection -ComputerName SERVEUR_QCLICKER -Port 443
```

Sortie utile :

```text
RemoteAddress    : 10.20.30.40
RemotePort       : 443
TcpTestSucceeded : True
```

`True` confirme que le PC atteint directement le port. `False` signifie que la
connexion directe échoue ; cela peut être normal si le réseau impose un proxy.

### Étape 7 — comparer curl direct et curl via proxy

Connexion forcée sans proxy :

```powershell
curl.exe -v --noproxy "*" --connect-timeout 30 --max-time 330 `
  "https://SERVEUR_QCLICKER/api/GetCustomers?page=1"
```

Si l'étape 2 a donné un proxy manuel, tester le même appel via ce proxy :

```powershell
curl.exe -v --proxy "http://proxy.entreprise.local:8080" `
  --connect-timeout 30 --max-time 330 `
  "https://SERVEUR_QCLICKER/api/GetCustomers?page=1"
```

Repères dans la sortie `-v` :

| Ligne curl | Interprétation |
|---|---|
| `Trying 10.x.x.x:443` | curl tente une connexion directe au serveur. |
| `Connected to ...` | TCP est établi. |
| `CONNECT SERVEUR_QCLICKER:443` | curl passe par un proxy HTTP. |
| `HTTP/1.1 200 Connection established` | Le proxy a créé le tunnel HTTPS. |
| `407 Proxy Authentication Required` | Le proxy réclame une authentification. |
| `SSL certificate problem` | Le réseau est atteint ; le certificat doit être résolu, pas ignoré durablement. |

### Décision finale

| Observation | Configuration à utiliser dans FastAPI/HTTPX |
|---|---|
| Direct fonctionne | `use_system_proxy: false` ou aucun proxy configuré. |
| Seulement le proxy manuel fonctionne | Laisser `use_system_proxy: true` ; la passerelle lit `ProxyServer`. |
| Un PAC est le seul chemin qui fonctionne | Obtenir auprès du réseau le proxy résolu pour Qlicker et définir `QLICKER_PROXY_URL`. |
| Postman seul fonctionne encore | Comparer dans Postman l'URL, méthode, query params, headers, cookies, certificat client et proxy personnalisé. |

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
