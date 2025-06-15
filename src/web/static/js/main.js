// Main JavaScript for Horse Racing Prediction web application

// Global variables
let updateInterval;
let lastUpdateTime = new Date();
let currentCountryFilter = 'All';
let currentTimeFilter = 'all';

// DOM ready
document.addEventListener('DOMContentLoaded', function() {
    // Initialize the application
    initApp();
    
    // Set up event listeners
    document.getElementById('apply-filters').addEventListener('click', applyFilters);
    
    // Set up navigation scroll
    setupSmoothScroll();
});

// Initialize the application
function initApp() {
    // Load initial data
    loadStats();
    loadRaces();
    loadRecommendations();
    
    // Set up auto-refresh (every 30 seconds)
    updateInterval = setInterval(function() {
        refreshData();
    }, 30000);
}

// Load model statistics
function loadStats() {
    fetch('/api/stats')
        .then(response => response.json())
        .then(data => {
            document.getElementById('win-rate').textContent = `${(data.win_rate * 100).toFixed(1)}%`;
            document.getElementById('roi').textContent = `${data.roi.toFixed(1)}%`;
            document.getElementById('confidence').textContent = `${(data.confidence * 100).toFixed(0)}%`;
            document.getElementById('races-analyzed').textContent = data.races_analyzed.toLocaleString();
            
            // Update strategy details
            const strategyHTML = `
                <ul class="list-unstyled mb-0">
                    <li><strong>Win Probability:</strong> ≥ ${data.optimal_strategy.threshold}</li>
                    <li><strong>Odds Range:</strong> ${data.optimal_strategy.min_odds} to ${data.optimal_strategy.max_odds}</li>
                    <li><strong>Expected Value:</strong> ≥ ${data.optimal_strategy.min_ev}</li>
                </ul>
            `;
            document.getElementById('strategy-details').innerHTML = strategyHTML;
        })
        .catch(error => {
            console.error('Error loading stats:', error);
        });
}

// Load upcoming races
function loadRaces() {
    // Show loading indicator
    updateLoadingStatus(true);
    
    fetch(`/api/races?country=${currentCountryFilter}`)
        .then(response => response.json())
        .then(races => {
            // Filter races by time if needed
            const filteredRaces = filterRacesByTime(races, currentTimeFilter);
            
            // Update the races table
            updateRacesTable(filteredRaces);
            
            // Update loading status
            updateLoadingStatus(false);
        })
        .catch(error => {
            console.error('Error loading races:', error);
            updateLoadingStatus(false);
        });
}

// Load betting recommendations
function loadRecommendations() {
    fetch('/api/recommendations')
        .then(response => response.json())
        .then(recommendations => {
            updateRecommendationsTable(recommendations);
        })
        .catch(error => {
            console.error('Error loading recommendations:', error);
        });
}

// Update races table with data
function updateRacesTable(races) {
    const tableBody = document.getElementById('races-table');
    
    if (races.length === 0) {
        tableBody.innerHTML = '<tr><td colspan="9" class="text-center">No races found matching your filters</td></tr>';
        return;
    }
    
    let html = '';
    
    races.forEach(race => {
        const raceTime = new Date(race.race_time);
        const timeString = formatDateTime(raceTime);
        const countdown = getCountdown(raceTime);
        
        html += `
            <tr>
                <td>${race.race_name}</td>
                <td>${race.track}</td>
                <td>${race.country}</td>
                <td>
                    ${timeString}
                    <div class="race-countdown">${countdown}</div>
                </td>
                <td>${race.race_class}</td>
                <td>${race.distance}m</td>
                <td>${race.going}</td>
                <td>${race.num_runners}</td>
                <td>
                    <button class="btn btn-sm btn-primary view-race" data-race-id="${race.race_id}" 
                            data-bs-toggle="modal" data-bs-target="#raceModal">
                        <i class="fas fa-eye"></i> View
                    </button>
                </td>
            </tr>
        `;
    });
    
    tableBody.innerHTML = html;
    
    // Add event listeners to view buttons
    document.querySelectorAll('.view-race').forEach(button => {
        button.addEventListener('click', function() {
            const raceId = this.getAttribute('data-race-id');
            loadRaceDetails(raceId);
        });
    });
}

// Update recommendations table with data
function updateRecommendationsTable(recommendations) {
    const tableBody = document.getElementById('recommendations-table');
    
    if (recommendations.length === 0) {
        tableBody.innerHTML = '<tr><td colspan="7" class="text-center">No recommendations available at this time</td></tr>';
        return;
    }
    
    let html = '';
    
    // Limit to top 10 recommendations
    const topRecommendations = recommendations.slice(0, 10);
    
    topRecommendations.forEach(rec => {
        const raceTime = new Date(rec.race_time);
        const timeString = formatDateTime(raceTime);
        const probClass = getProbabilityClass(rec.win_probability);
        const evClass = rec.expected_value > 0 ? 'ev-positive' : 'ev-negative';
        
        html += `
            <tr>
                <td>${rec.race_name} (${rec.country})</td>
                <td>${timeString}</td>
                <td>${rec.horse_name}</td>
                <td>${rec.odds.toFixed(2)}</td>
                <td class="${probClass}">${(rec.win_probability * 100).toFixed(1)}%</td>
                <td class="${evClass}">${rec.expected_value.toFixed(2)}</td>
                <td>
                    <button class="btn btn-sm btn-primary view-race" data-race-id="${rec.race_id}" 
                            data-bs-toggle="modal" data-bs-target="#raceModal">
                        <i class="fas fa-eye"></i> View Race
                    </button>
                </td>
            </tr>
        `;
    });
    
    tableBody.innerHTML = html;
    
    // Add event listeners to view buttons
    document.querySelectorAll('.view-race').forEach(button => {
        button.addEventListener('click', function() {
            const raceId = this.getAttribute('data-race-id');
            loadRaceDetails(raceId);
        });
    });
}

// Load race details for the modal
function loadRaceDetails(raceId) {
    fetch(`/api/race/${raceId}`)
        .then(response => response.json())
        .then(race => {
            // Update modal title
            document.getElementById('raceModalTitle').textContent = `${race.race_name} - ${race.track}, ${race.country}`;
            
            // Update race info
            const raceTime = new Date(race.race_time);
            const raceInfoHTML = `
                <h5>Race Information</h5>
                <ul class="list-unstyled">
                    <li><strong>Date & Time:</strong> ${formatDateTime(raceTime)}</li>
                    <li><strong>Track:</strong> ${race.track}, ${race.country}</li>
                    <li><strong>Class:</strong> ${race.race_class}</li>
                    <li><strong>Distance:</strong> ${race.distance}m</li>
                    <li><strong>Going:</strong> ${race.going}</li>
                    <li><strong>Runners:</strong> ${race.num_runners}</li>
                </ul>
            `;
            document.getElementById('race-info').innerHTML = raceInfoHTML;
            
            // Update race stats
            const recommendedHorses = race.horses.filter(h => h.recommended).length;
            const raceStatsHTML = `
                <ul class="list-unstyled">
                    <li><strong>Recommended Bets:</strong> ${recommendedHorses}</li>
                    <li><strong>Favorite:</strong> ${race.horses[0].horse_name} (${race.horses[0].odds.toFixed(2)})</li>
                    <li><strong>Highest Win Probability:</strong> ${(race.horses[0].win_probability * 100).toFixed(1)}%</li>
                    <li><strong>Average Odds:</strong> ${calculateAverageOdds(race.horses).toFixed(2)}</li>
                </ul>
            `;
            document.getElementById('race-stats').innerHTML = raceStatsHTML;
            
            // Update horses table
            updateRaceHorsesTable(race.horses);
        })
        .catch(error => {
            console.error('Error loading race details:', error);
        });
}

// Update horses table in race modal
function updateRaceHorsesTable(horses) {
    const tableBody = document.getElementById('race-horses');
    
    let html = '';
    
    // Sort horses by win probability (highest first)
    horses.sort((a, b) => b.win_probability - a.win_probability);
    
    horses.forEach(horse => {
        const probClass = getProbabilityClass(horse.win_probability);
        const evClass = horse.expected_value > 0 ? 'ev-positive' : 'ev-negative';
        const rowClass = horse.recommended ? 'recommended' : '';
        
        html += `
            <tr class="${rowClass}">
                <td>${horse.horse_name}</td>
                <td>${horse.age}</td>
                <td>${horse.jockey}</td>
                <td>${horse.trainer}</td>
                <td>${horse.weight.toFixed(1)}kg</td>
                <td>${horse.draw}</td>
                <td>${horse.odds.toFixed(2)}</td>
                <td class="${probClass}">${(horse.win_probability * 100).toFixed(1)}%</td>
                <td class="${evClass}">${horse.expected_value.toFixed(2)}</td>
                <td>
                    ${horse.recommended ? 
                        '<span class="badge-recommendation"><i class="fas fa-check"></i> Recommended</span>' : 
                        '<span class="text-muted">Not recommended</span>'}
                </td>
            </tr>
        `;
    });
    
    tableBody.innerHTML = html;
}

// Apply filters when button is clicked
function applyFilters() {
    currentCountryFilter = document.getElementById('country-filter').value;
    currentTimeFilter = document.getElementById('time-filter').value;
    
    loadRaces();
    // Also refresh recommendations as they might be affected by filters
    loadRecommendations();
}

// Filter races by time frame
function filterRacesByTime(races, timeFilter) {
    const now = new Date();
    
    switch(timeFilter) {
        case 'today':
            return races.filter(race => {
                const raceTime = new Date(race.race_time);
                return raceTime.toDateString() === now.toDateString();
            });
        case 'hour':
            return races.filter(race => {
                const raceTime = new Date(race.race_time);
                return (raceTime - now) <= 60 * 60 * 1000; // 1 hour in milliseconds
            });
        case '6hours':
            return races.filter(race => {
                const raceTime = new Date(race.race_time);
                return (raceTime - now) <= 6 * 60 * 60 * 1000; // 6 hours in milliseconds
            });
        case 'all':
        default:
            return races;
    }
}

// Refresh all data
function refreshData() {
    updateLoadingStatus(true);
    
    // Refresh all data sources
    loadStats();
    loadRaces();
    loadRecommendations();
    
    // Update last update time
    lastUpdateTime = new Date();
    document.getElementById('last-update').textContent = `Last update: ${formatTime(lastUpdateTime)}`;
    
    updateLoadingStatus(false);
}

// Update loading status indicator
function updateLoadingStatus(isLoading) {
    const indicator = document.getElementById('update-status');
    
    if (isLoading) {
        indicator.classList.add('updating');
    } else {
        indicator.classList.remove('updating');
    }
}

// Set up smooth scrolling for navigation
function setupSmoothScroll() {
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function(e) {
            e.preventDefault();
            
            const targetId = this.getAttribute('href');
            if (targetId === '#') return;
            
            const targetElement = document.querySelector(targetId);
            if (targetElement) {
                targetElement.scrollIntoView({
                    behavior: 'smooth',
                    block: 'start'
                });
            }
        });
    });
}

// Helper function to format date and time
function formatDateTime(date) {
    return date.toLocaleString('en-US', {
        weekday: 'short',
        month: 'short',
        day: 'numeric',
        hour: '2-digit',
        minute: '2-digit'
    });
}

// Helper function to format time only
function formatTime(date) {
    return date.toLocaleTimeString('en-US', {
        hour: '2-digit',
        minute: '2-digit',
        second: '2-digit'
    });
}

// Helper function to get countdown string
function getCountdown(raceTime) {
    const now = new Date();
    const diff = raceTime - now;
    
    if (diff <= 0) {
        return "Started";
    }
    
    const hours = Math.floor(diff / (1000 * 60 * 60));
    const minutes = Math.floor((diff % (1000 * 60 * 60)) / (1000 * 60));
    
    if (hours > 0) {
        return `${hours}h ${minutes}m`;
    } else {
        return `${minutes}m`;
    }
}

// Helper function to get probability class
function getProbabilityClass(probability) {
    if (probability >= 0.6) {
        return 'prob-high';
    } else if (probability >= 0.3) {
        return 'prob-medium';
    } else {
        return 'prob-low';
    }
}

// Helper function to calculate average odds
function calculateAverageOdds(horses) {
    const sum = horses.reduce((total, horse) => total + horse.odds, 0);
    return sum / horses.length;
}
