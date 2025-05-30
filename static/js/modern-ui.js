/**
 * Script pour améliorer l'interface utilisateur moderne de GeminiChat
 */

document.addEventListener('DOMContentLoaded', function() {
    // Animation des nœuds de réseau neural
    animateNeuralNetwork();
    
    // Gestion du menu mobile
    initializeMobileMenu();
    
    // Animation au défilement
    initializeScrollAnimations();
});

/**
 * Anime les nœuds du réseau neural avec des connexions
 */
function animateNeuralNetwork() {
    const neuralNetwork = document.querySelector('.neural-network');
    if (!neuralNetwork) return;
    
    // Ajouter les connexions entre les nœuds
    const nodes = neuralNetwork.querySelectorAll('.node');
    
    for (let i = 0; i < nodes.length; i++) {
        for (let j = i + 1; j < nodes.length; j++) {
            if (Math.random() > 0.5) { // Ajouter des connexions aléatoires
                const connection = document.createElement('div');
                connection.className = 'connection';
                
                // Calculer les positions pour la ligne
                const node1Rect = nodes[i].getBoundingClientRect();
                const node2Rect = nodes[j].getBoundingClientRect();
                const neuralRect = neuralNetwork.getBoundingClientRect();
                
                const x1 = (node1Rect.left + node1Rect.width/2) - neuralRect.left;
                const y1 = (node1Rect.top + node1Rect.height/2) - neuralRect.top;
                const x2 = (node2Rect.left + node2Rect.width/2) - neuralRect.left;
                const y2 = (node2Rect.top + node2Rect.height/2) - neuralRect.top;
                
                // Définir la ligne
                connection.style.position = 'absolute';
                connection.style.height = '2px';
                connection.style.width = Math.sqrt(Math.pow(x2-x1, 2) + Math.pow(y2-y1, 2)) + 'px';
                connection.style.backgroundColor = 'rgba(139, 92, 246, 0.3)';
                connection.style.transformOrigin = '0 0';
                connection.style.left = x1 + 'px';
                connection.style.top = y1 + 'px';
                connection.style.transform = `rotate(${Math.atan2(y2-y1, x2-x1)}rad)`;
                
                // Animation de pulsation
                connection.style.animation = `pulse-connection 3s infinite ${Math.random()}s`;
                
                neuralNetwork.appendChild(connection);
            }
        }
    }
}

/**
 * Initialise le menu mobile
 */
function initializeMobileMenu() {
    const mobileMenuToggle = document.getElementById('mobile-menu-toggle');
    const mobileMenuClose = document.getElementById('mobile-menu-close');
    const mobileMenu = document.getElementById('mobile-menu');
    
    if (!mobileMenuToggle || !mobileMenuClose || !mobileMenu) return;
    
    mobileMenuToggle.addEventListener('click', function() {
        mobileMenu.classList.add('active');
        document.body.style.overflow = 'hidden';
    });
    
    mobileMenuClose.addEventListener('click', function() {
        mobileMenu.classList.remove('active');
        document.body.style.overflow = '';
    });
}

/**
 * Initialise les animations au défilement
 */
function initializeScrollAnimations() {
    const animatedElements = document.querySelectorAll('.feature-card, .info-content, .info-image, .testimonial-card');
    
    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.classList.add('animated');
                observer.unobserve(entry.target);
            }
        });
    }, { threshold: 0.1 });
    
    animatedElements.forEach(el => {
        el.classList.add('fade-in');
        observer.observe(el);
    });
}
