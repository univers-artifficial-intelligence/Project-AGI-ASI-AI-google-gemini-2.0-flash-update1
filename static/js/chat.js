document.addEventListener('DOMContentLoaded', function() {
    const chatMessages = document.getElementById('chat-messages');
    const userInput = document.getElementById('user-input');
    const sendButton = document.getElementById('send-button');
    const imageUpload = document.getElementById('image-upload');
    const imagePreviewContainer = document.getElementById('image-preview-container');
    const documentUpload = document.getElementById('document-upload');
    const filePreviewContainer = document.getElementById('file-preview-container');
    const emotionValue = document.getElementById('emotion-value');
    
    let currentImageData = null;
    let currentFileData = null;
    
    // Fonction pour mettre à jour l'affichage de l'état émotionnel
    function updateEmotionalStateDisplay(emotionalState) {
        if (!emotionalState || !emotionValue) return;
        
        // Mettre à jour l'étiquette d'émotion uniquement
        emotionValue.textContent = emotionalState.base_state || 'neutre';
    }

    // Fonction pour ajouter un message à la conversation
    function addMessage(content, isUser = false, imageData = null, fileData = null, emotionalState = null) {
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${isUser ? 'user' : 'bot'}`;
        
        const messageContent = document.createElement('div');
        messageContent.className = 'message-content';
        
        // Si c'est un message de l'IA et qu'on a un état émotionnel significatif
        if (!isUser && emotionalState && emotionalState.base_state && 
            !['calm', 'neutral'].includes(emotionalState.base_state.toLowerCase())) {
            const emotionTag = document.createElement('span');
            emotionTag.className = 'emotion-tag';
            emotionTag.dataset.emotion = emotionalState.base_state; // Ajouter l'attribut data pour le CSS
            emotionTag.textContent = emotionalState.base_state;
            messageContent.appendChild(emotionTag);
        }
        
        // Si un message texte est présent
        if (content) {
            // Créer un élément pour le texte du message
            const textElem = document.createElement('span');
            textElem.textContent = content;
            messageContent.appendChild(textElem);
        }
        
        // Si une image est présente
        if (imageData && isUser) {
            const imageElement = document.createElement('img');
            imageElement.src = imageData;
            imageElement.className = 'message-image';
            messageContent.appendChild(imageElement);
        }
        
        // Si un fichier est présent
        if (fileData && isUser) {
            const fileAttachment = document.createElement('div');
            fileAttachment.className = 'file-attachment';
            
            const fileIcon = document.createElement('i');
            fileIcon.className = `fas ${getFileIcon(fileData.file_type)}`;
            fileAttachment.appendChild(fileIcon);
            
            const fileInfo = document.createElement('div');
            fileInfo.className = 'file-attachment-info';
            fileInfo.textContent = `${fileData.filename} (${fileData.file_size})`;
            fileAttachment.appendChild(fileInfo);
            
            messageContent.appendChild(fileAttachment);
        }
        
        messageDiv.appendChild(messageContent);
        chatMessages.appendChild(messageDiv);
        
        // Scroll vers le bas pour voir le nouveau message
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }
    
    // Fonction pour afficher l'animation d'écriture
    function showTypingIndicator() {
        const typingDiv = document.createElement('div');
        typingDiv.className = 'thinking-animation';
        typingDiv.id = 'typing-indicator';
        
        const dotsContainer = document.createElement('div');
        dotsContainer.className = 'thinking-dots';
        
        // Ajouter 3 points d'animation
        for (let i = 0; i < 3; i++) {
            const dot = document.createElement('div');
            dot.className = 'thinking-dot';
            dotsContainer.appendChild(dot);
        }
        
        const textSpan = document.createElement('span');
        textSpan.className = 'thinking-text';
        textSpan.textContent = 'Gemini réfléchit';
        
        typingDiv.appendChild(dotsContainer);
        typingDiv.appendChild(textSpan);
        
        chatMessages.appendChild(typingDiv);
        chatMessages.scrollTop = chatMessages.scrollHeight;
        
        return typingDiv;
    }
    
    // Fonction pour supprimer l'animation d'écriture
    function removeTypingIndicator() {
        const typingIndicator = document.getElementById('typing-indicator');
        if (typingIndicator) {
            chatMessages.removeChild(typingIndicator);
        }
    }
    
    // Fonction pour déterminer l'icône de fichier en fonction du type
    function getFileIcon(fileType) {
        const icons = {
            'PDF': 'fa-file-pdf',
            'Word': 'fa-file-word',
            'Text': 'fa-file-alt',
            'CSV': 'fa-file-csv',
            'Excel': 'fa-file-excel',
            'JSON': 'fa-file-code',
            'HTML': 'fa-file-code',
            'XML': 'fa-file-code',
            'Markdown': 'fa-file-alt',
            'Rich Text': 'fa-file-alt'
        };
        
        return icons[fileType] || 'fa-file';
    }

    // Fonction pour récupérer l'état émotionnel actuel
    async function fetchEmotionalState() {
        try {
            const response = await fetch('/api/emotional_state', {
                method: 'GET',
                headers: {
                    'Content-Type': 'application/json',
                }
            });
            
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            
            const data = await response.json();
            return data;
        } catch (error) {
            console.error('Error fetching emotional state:', error);
            return null;
        }
    }

    // Fonction pour envoyer un message à l'API
    async function sendMessage(message, imageData = null, fileData = null) {
        try {
            const payload = { message };
            
            // Ajouter l'image en base64 si présente
            if (imageData) {
                payload.image = imageData;
                console.log("Image ajoutée à la requête");
            }
            
            // Ajouter l'ID du fichier si présent
            if (fileData) {
                payload.file_id = fileData.id;
            }
            
            // Ajouter l'ID de session si disponible via l'URL
            const urlParams = new URLSearchParams(window.location.search);
            const sessionId = urlParams.get('session_id');
            if (sessionId) {
                payload.session_id = sessionId;
            }
            
            // Ajouter le fuseau horaire de l'utilisateur
            payload.user_timezone = Intl.DateTimeFormat().resolvedOptions().timeZone;
            
            console.log("Envoi de la requête à l'API...");
            
            const response = await fetch('/api/chat', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(payload),
            });
            
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            
            const data = await response.json();
            console.log("Réponse reçue de l'API");
            
            // Si l'état émotionnel est inclus dans la réponse, le mettre à jour
            if (data.emotional_state) {
                updateEmotionalStateDisplay(data.emotional_state);
            }
            
            // Si un ID de session est retourné et que nous n'avons pas d'ID de session dans l'URL
            if (data.session_id && !sessionId) {
                // Mettre à jour l'URL avec le nouvel ID de session sans recharger la page
                const newUrl = new URL(window.location);
                newUrl.searchParams.set('session_id', data.session_id);
                window.history.pushState({}, '', newUrl);
            }
            
            return {
                response: data.response,
                emotionalState: data.emotional_state
            };
        } catch (error) {
            console.error('Error:', error);
            return {
                response: "Désolé, une erreur s'est produite lors de la communication avec Gemini."
            };
        }
    }
    
    // Gestionnaire d'événement pour l'envoi de message
    async function handleSend() {
        const message = userInput.value.trim();
        if (!message && !currentImageData && !currentFileData) return;
        
        // Afficher le message de l'utilisateur
        addMessage(message, true, currentImageData, currentFileData);
        
        // Sauvegarder une copie des données actuelles
        const sentImageData = currentImageData;
        const sentFileData = currentFileData;
        
        // Effacer l'entrée utilisateur et réinitialiser l'image/fichier
        userInput.value = '';
        currentImageData = null;
        imagePreviewContainer.innerHTML = '';
        imageUpload.value = '';
        currentFileData = null;
        filePreviewContainer.innerHTML = '';
        documentUpload.value = '';
        
        // Afficher l'animation d'écriture moderne
        showTypingIndicator();
        
        // Envoyer le message à l'API
        const result = await sendMessage(message, sentImageData, sentFileData);
        
        // Supprimer l'indicateur de chargement
        removeTypingIndicator();
        
        // Afficher la réponse avec l'état émotionnel
        addMessage(result.response, false, null, null, result.emotionalState);
    }

    // Attachement des événements
    sendButton.addEventListener('click', handleSend);
    
    userInput.addEventListener('keypress', function(e) {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            handleSend();
        }
    });
    
    // Gestion de l'upload d'image
    imageUpload.addEventListener('change', function(e) {
        const file = e.target.files[0];
        if (!file) return;
        
        // Vérifier si c'est bien une image
        if (!file.type.match('image.*')) {
            alert('Veuillez sélectionner une image.');
            return;
        }
        
        const reader = new FileReader();
        
        reader.onload = function(e) {
            currentImageData = e.target.result; // Image en base64
            console.log("Image convertie en base64");
            
            // Afficher la prévisualisation
            imagePreviewContainer.innerHTML = '';
            const previewElement = document.createElement('div');
            previewElement.className = 'image-preview';
            
            const img = document.createElement('img');
            img.src = currentImageData;
            
            const removeBtn = document.createElement('button');
            removeBtn.className = 'remove-image-btn';
            removeBtn.innerHTML = '<i class="fas fa-times"></i>';
            removeBtn.addEventListener('click', function() {
                currentImageData = null;
                imagePreviewContainer.innerHTML = '';
                imageUpload.value = '';
            });
            
            previewElement.appendChild(img);
            previewElement.appendChild(removeBtn);
            imagePreviewContainer.appendChild(previewElement);
            
            // Réinitialiser le fichier si une image est sélectionnée
            if (currentFileData) {
                currentFileData = null;
                filePreviewContainer.innerHTML = '';
                documentUpload.value = '';
            }
        };
        
        reader.readAsDataURL(file);
    });
    
    // Gestion de l'upload de document
    documentUpload.addEventListener('change', function(e) {
        const file = e.target.files[0];
        if (!file) return;
        
        // Vérifier l'extension du fichier
        const fileExt = file.name.split('.').pop().toLowerCase();
        const allowedExts = ['pdf', 'doc', 'docx', 'txt', 'csv', 'xls', 'xlsx', 'json', 'html', 'xml', 'md', 'rtf'];
        
        if (!allowedExts.includes(fileExt)) {
            alert('Type de fichier non pris en charge.');
            return;
        }
        
        // Créer un FormData pour l'upload
        const formData = new FormData();
        formData.append('file', file);
        
        // Afficher un indicateur de chargement
        filePreviewContainer.innerHTML = '<div class="file-loading">Chargement du fichier...</div>';
        
        // Envoyer le fichier au serveur
        fetch('/api/upload-file', {
            method: 'POST',
            body: formData
        })
        .then(response => {
            if (!response.ok) {
                throw new Error(`Erreur HTTP: ${response.status}`);
            }
            return response.json();
        })
        .then(data => {
            // Stocker les données du fichier
            currentFileData = data;
            
            // Afficher la prévisualisation du fichier
            filePreviewContainer.innerHTML = '';
            const previewElement = document.createElement('div');
            previewElement.className = 'file-preview';
            
            const fileIcon = document.createElement('i');
            fileIcon.className = `fas ${getFileIcon(data.file_type)}`;
            previewElement.appendChild(fileIcon);
            
            const fileInfo = document.createElement('div');
            fileInfo.className = 'file-info';
            
            const fileName = document.createElement('div');
            fileName.className = 'file-name';
            fileName.textContent = data.original_filename;
            fileInfo.appendChild(fileName);
            
            const fileSize = document.createElement('div');
            fileSize.className = 'file-size';
            fileSize.textContent = data.file_size;
            fileInfo.appendChild(fileSize);
            
            previewElement.appendChild(fileInfo);
            
            const removeBtn = document.createElement('button');
            removeBtn.className = 'remove-file-btn';
            removeBtn.innerHTML = '<i class="fas fa-times"></i>';
            removeBtn.addEventListener('click', function() {
                currentFileData = null;
                filePreviewContainer.innerHTML = '';
                documentUpload.value = '';
            });
            
            previewElement.appendChild(removeBtn);
            filePreviewContainer.appendChild(previewElement);
            
            // Réinitialiser l'image si un fichier est sélectionné
            if (currentImageData) {
                currentImageData = null;
                imagePreviewContainer.innerHTML = '';
                imageUpload.value = '';
            }
        })
        .catch(error => {
            console.error('Erreur:', error);
            filePreviewContainer.innerHTML = `<div class="file-error">Erreur: ${error.message}</div>`;
        });
    });
    
    // Fonction pour optimiser l'espacement des messages longs sur mobile
    function optimizeMessageSpacing() {
        // Vérifier si on est sur mobile
        if (window.innerWidth <= 768) {
            // Cibler tous les messages
            const messages = document.querySelectorAll('.message');
            
            messages.forEach(msg => {
                // Si le contenu est long (+ de 200 caractères), ajouter une classe spéciale
                if (msg.textContent.length > 200) {
                    msg.classList.add('long-content');
                }
                
                // Améliorer l'espacement dans les listes
                const lists = msg.querySelectorAll('ul, ol');
                lists.forEach(list => {
                    list.style.marginTop = '1rem';
                    list.style.marginBottom = '1rem';
                });
                
                // Améliorer l'espacement des paragraphes
                const paragraphs = msg.querySelectorAll('p');
                if (paragraphs.length > 3) {
                    paragraphs.forEach(p => {
                        p.style.marginBottom = '1.2rem';
                    });
                }
            });
        }
    }
    
    // Exécuter la fonction au chargement et au resize
    optimizeMessageSpacing();
    window.addEventListener('resize', optimizeMessageSpacing);
    
    // Observer les nouveaux messages pour leur appliquer l'optimisation
    const observer = new MutationObserver(function(mutations) {
        mutations.forEach(function(mutation) {
            if (mutation.type === 'childList' && mutation.addedNodes.length > 0) {
                optimizeMessageSpacing();
            }
        });
    });
    
    // Observer les changements dans le conteneur de messages
    if (chatMessages) {
        observer.observe(chatMessages, { childList: true });
    }
    
    // Exposer des fonctions pour être utilisées par d'autres scripts
    window.addMessage = addMessage;
    window.getFileIcon = getFileIcon;
    window.updateEmotionalStateDisplay = updateEmotionalStateDisplay;
    window.sendMessage = sendMessage;
    window.showTypingIndicator = showTypingIndicator;
    window.removeTypingIndicator = removeTypingIndicator;
});
