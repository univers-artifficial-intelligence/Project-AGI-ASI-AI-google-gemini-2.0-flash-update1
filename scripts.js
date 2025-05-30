// Simple script to handle scroll animations and navigation

document.addEventListener('DOMContentLoaded', () => {
    // Navbar scroll effect
    const nav = document.querySelector('nav');
    
    window.addEventListener('scroll', () => {
        if (window.scrollY > 100) {
            nav.style.padding = '15px 40px';
            nav.style.boxShadow = '0 10px 30px rgba(0,0,0,0.1)';
        } else {
            nav.style.padding = '20px 40px';
            nav.style.boxShadow = '0 5px 20px rgba(0,0,0,0.05)';
        }
    });

    // Mobile menu toggle
    const mobileNavToggle = document.querySelector('.mobile-nav-toggle');
    const mobileNav = document.querySelector('.nav-mobile');
    const mobileNavLinks = document.querySelectorAll('.nav-mobile a');
    
    if (mobileNavToggle && mobileNav) {
        mobileNavToggle.addEventListener('click', () => {
            mobileNav.classList.toggle('active');
            
            // Change icon based on menu state
            const icon = mobileNavToggle.querySelector('i');
            if (mobileNav.classList.contains('active')) {
                icon.classList.remove('fa-bars');
                icon.classList.add('fa-times');
            } else {
                icon.classList.remove('fa-times');
                icon.classList.add('fa-bars');
            }
        });
        
        // Close mobile menu when a link is clicked
        mobileNavLinks.forEach(link => {
            link.addEventListener('click', () => {
                mobileNav.classList.remove('active');
                const icon = mobileNavToggle.querySelector('i');
                icon.classList.remove('fa-times');
                icon.classList.add('fa-bars');
            });
        });
    }

    // Smooth scrolling for anchor links
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function (e) {
            e.preventDefault();
            
            const targetId = this.getAttribute('href');
            const targetElement = document.querySelector(targetId);
            
            if (targetElement) {
                window.scrollTo({
                    top: targetElement.offsetTop - 80,
                    behavior: 'smooth'
                });
            }
        });
    });

    // Animation for elements when they enter the viewport
    const animatedElements = document.querySelectorAll('.feature-card, .capability-item, .tech-step');
    
    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.style.opacity = '1';
                entry.target.style.transform = entry.target.classList.contains('tech-step') 
                    ? 'translateX(0)' 
                    : 'translateY(0)';
                observer.unobserve(entry.target);
            }
        });
    }, { threshold: 0.1 });
    
    animatedElements.forEach(el => {
        el.style.opacity = '0';
        el.style.transform = el.classList.contains('tech-step') 
            ? 'translateX(-20px)' 
            : 'translateY(20px)';
        el.style.transition = 'opacity 0.6s ease, transform 0.6s ease';
        observer.observe(el);
    });
    
    // Particle effect for hero section
    const createParticles = () => {
        const heroSection = document.querySelector('.hero');
        if (!heroSection) return;
        
        for (let i = 0; i < 30; i++) {
            const particle = document.createElement('div');
            particle.classList.add('particle');
            
            const size = Math.random() * 5 + 1;
            const posX = Math.random() * 100;
            const posY = Math.random() * 100;
            const duration = Math.random() * 20 + 10;
            const delay = Math.random() * 5;
            
            particle.style.width = `${size}px`;
            particle.style.height = `${size}px`;
            particle.style.left = `${posX}%`;
            particle.style.top = `${posY}%`;
            particle.style.animationDuration = `${duration}s`;
            particle.style.animationDelay = `${delay}s`;
            
            heroSection.appendChild(particle);
        }
    };
    
    createParticles();

    // Amélioration de l'affichage du titre sur mobile
    optimizeTitleDisplay();
    
    // Gestion du redimensionnement de fenêtre
    window.addEventListener('resize', optimizeTitleDisplay);
    
    // Initialisation des modals
    initializeModals();
    
    // Optimisation mobile
    if (window.innerWidth <= 767) {
        optimizeMobileInterface();
    }
});

// Fonction pour gérer l'orientation mobile
window.addEventListener('orientationchange', function() {
    setTimeout(() => {
        optimizeTitleDisplay();
        optimizeMobileInterface();
    }, 300);
});

// Améliorer l'accessibilité
document.addEventListener('keydown', function(e) {
    // Fermer les modals avec Escape
    if (e.key === 'Escape') {
        const openModal = document.querySelector('.modal[style*="display: flex"]');
        if (openModal) {
            openModal.style.display = 'none';
        }
    }
});

function optimizeTitleDisplay() {
    const titleElement = document.querySelector('.app-title');
    const titleMain = document.querySelector('.title-main');
    const titleSub = document.querySelector('.title-sub');
    
    if (!titleElement || !titleMain || !titleSub) return;
    
    const containerWidth = titleElement.parentElement.offsetWidth;
    const isMobile = window.innerWidth <= 767;
    
    if (isMobile) {
        // Mode mobile : affichage en colonne
        titleElement.style.flexDirection = 'column';
        titleElement.style.alignItems = 'center';
        titleMain.style.marginRight = '0';
        titleMain.style.marginBottom = '2px';
        
        // Ajuster la taille de police si nécessaire
        if (containerWidth < 320) {
            titleMain.style.fontSize = '1.2rem';
            titleSub.style.fontSize = '0.9rem';
        } else if (containerWidth < 400) {
            titleMain.style.fontSize = '1.3rem';
            titleSub.style.fontSize = '1rem';
        } else {
            titleMain.style.fontSize = '1.4rem';
            titleSub.style.fontSize = '1.1rem';
        }
    } else {
        // Mode desktop : affichage en ligne
        titleElement.style.flexDirection = 'row';
        titleElement.style.alignItems = 'baseline';
        titleMain.style.marginRight = '8px';
        titleMain.style.marginBottom = '0';
        titleMain.style.fontSize = '1.8rem';
        titleSub.style.fontSize = '1.3rem';
    }
}

function optimizeMobileInterface() {
    // Optimiser l'interface pour mobile
    const chatContainer = document.querySelector('.chat-container');
    if (chatContainer) {
        chatContainer.style.borderRadius = '0';
        chatContainer.style.margin = '0';
        chatContainer.style.maxWidth = '100%';
    }
    
    // Ajuster la hauteur des messages
    const chatMessages = document.querySelector('.chat-messages');
    if (chatMessages) {
        const viewportHeight = window.innerHeight;
        const headerHeight = document.querySelector('header')?.offsetHeight || 60;
        const inputHeight = document.querySelector('.chat-input')?.offsetHeight || 100;
        const compactNavHeight = 50;
        
        const availableHeight = viewportHeight - headerHeight - inputHeight - compactNavHeight - 20;
        chatMessages.style.height = Math.max(300, availableHeight) + 'px';
    }
}

function initializeModals() {
    // Gestion des modals
    const modals = document.querySelectorAll('.modal');
    
    modals.forEach(modal => {
        const closeBtn = modal.querySelector('.close-modal');
        if (closeBtn) {
            closeBtn.addEventListener('click', () => {
                modal.style.display = 'none';
            });
        }
        
        // Fermer en cliquant à l'extérieur
        modal.addEventListener('click', (e) => {
            if (e.target === modal) {
                modal.style.display = 'none';
            }
        });
    });
}

// Fonction utilitaire pour débouncer les événements
function debounce(func, wait) {
    let timeout;
    return function executedFunction(...args) {
        const later = () => {
            clearTimeout(timeout);
            func(...args);
        };
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
    };
}

// Optimiser le redimensionnement avec debounce
window.addEventListener('resize', debounce(function() {
    optimizeTitleDisplay();
    if (window.innerWidth <= 767) {
        optimizeMobileInterface();
    }
}, 250));
